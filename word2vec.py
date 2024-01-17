import argparse
import os,sys,math
import json
import yaml
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR

from models.cbow import CBOW_Model, SkipGram_Model
from models.create_model import initialize_weight
import torch.nn.functional as F

from utils.create_optim import create_optimizer

CBOW_N_WORDS = 4
SKIPGRAM_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 256

class Trainer:
    """Main class for model training"""
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        vocab,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.vocab=vocab
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train(self):
        initial_params = [param.clone() for param in self.model.parameters()]
        evaluate_word_analogy(self.model.input_layer.weight.data, self.vocab, analogy_dataset)
        for epoch in range(self.epochs):
            self._train_epoch(epoch, initial_params)
            self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )
            acc = evaluate_word_analogy(self.model.input_layer.weight.data, self.vocab, analogy_dataset)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
            if args.wandb:
                log = { 'epoch': epoch+1,
                        'iteration': (epoch+1) * len(self.train_dataloader),
                        'train_loss': self.loss["train"][-1],
                        'val_loss': self.loss["val"][-1],
                        'Analogy_Acc': 100*acc}
                wandb.log(log)
            if math.isnan(self.loss["val"][-1]) or self.loss["val"][-1]>20:
                print('Error: Train loss is nan', file=sys.stderr)
                sys.exit(0)

    def _train_epoch(self,epoch, initial_params):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # dummy_y = self.grad_maker.setup_model_call(self.model, inputs)
            # self.grad_maker.setup_loss_call(self.criterion, dummy_y, labels)
            # y, loss = self.grad_maker.forward_and_backward()
            # self.optimizer.step()
            # loss = self.criterion(y,labels)

            running_loss.append(loss.item())

            if i == self.train_steps:
                break
            
            if i%10==0:
                print(f'Epoch:{epoch+1} Iter:{i}/{len(self.train_dataloader)} Loss:{float(loss)}')
            if i%50==0 and args.wandb:
                weight_norm = get_weight_norm(model=self.model)
                weight_delta = weight_change(self.model, initial_params)
                grad_norm = get_grad_norm(model=self.model)

                if args.log_activation:
                    pre_act = get_pre_act(model=self.model)
                
                log = { 'epoch': epoch+1,
                        'iteration': (epoch) * len(self.train_dataloader)+i,
                        'train_loss': float(loss),
                        'weight_delta': float(weight_delta),
                        'weight_norm/':weight_norm,
                        'grad_norm/':grad_norm,
                        'lr':self.optimizer.param_groups[0]['lr']}
                if args.log_activation:
                    log['pre_act/'] = pre_act
                wandb.log(log)

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
    
def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.
    """
    if args.lr_scheduler == 'Linear':
        lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max = total_epochs)
    elif args.lr_scheduler == 'ExponentialLR':
        lr_scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: args.lr * (0.95 ** epoch))
    elif args.lr_scheduler == 'Constant':
        lr_scheduler = None
    return lr_scheduler

def save_config(config: dict, model_dir: str):
    """Save config file to `model_dir` directory"""
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)
        
def save_vocab(vocab, model_dir: str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)

def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

def get_data_iterator(ds_name, ds_type, data_dir):
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    data_iter = to_map_style_dataset(data_iter)
    return data_iter

def build_vocab(data_iter, tokenizer):
    """Builds vocabulary from iterator"""
    
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=args.min_word_frequency,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(
    model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None
):

    data_iter = get_data_iterator(ds_name, ds_type, data_dir)
    tokenizer = get_english_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)
        
    text_pipeline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab

def train():
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=args.model_name,
        ds_name=args.dataset,
        ds_type="train",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        vocab=None,
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=args.model_name,
        ds_name=args.dataset,
        ds_type="valid",
        data_dir=args.data_dir,
        batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        vocab=vocab,
    )

    vocab_size = len(vocab.get_stoi())
    wandb.run.summary['vocab_size'] = vocab_size
    print(f"Vocabulary size: {vocab_size}")

    if args.model_name == "cbow":
        model = CBOW_Model(vocab_size=vocab_size, embed_dim = args.embed_dim, embed_max_norm=args.embed_max_norm, bias=args.bias)
    if args.model_name == "skipgram":
        model = SkipGram_Model(vocab_size=vocab_size, embed_dim = args.embed_dim, embed_max_norm=args.embed_max_norm, bias=args.bias)
    model = initialize_weight(  model,
                                b_input=args.b_input,
                                b_hidden=args.b_hidden,
                                b_output=args.b_output,
                                output_nonzero=not args.output_zero,
                                output_var_mult=args.output_var_mult,
                                width=args.embed_dim / args.base_width,
                                embed_std=args.embed_std)
    if args.loss_type == 'nll':
        criterion = nn.NLLLoss()
    if args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    if args.loss_type == 'mse':
        criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.log_activation:
        model = register_fhook(model)
    optimizer = create_optimizer(args, model, lr = args.lr, head_only=False)
    lr_scheduler = get_lr_scheduler(optimizer, args.epochs, verbose=True)
    
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        train_dataloader=train_dataloader,
        vocab= vocab,
        train_steps=args.train_steps,
        val_dataloader=val_dataloader,
        val_steps=args.val_steps,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=args.checkpoint_frequency,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=args.model_dir,
        model_name=args.model_name,
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, args.model_dir)
    save_config(config, args.model_dir)
    print("Model artifacts saved to folder:", args.model_dir)

    return model, vocab

def get_weight_norm(model):
    norm_dic={}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        norm_dic[name] = torch.abs(module.weight).mean(dtype=torch.float32)
    return norm_dic

def get_grad_norm(model):
    norm_dic={}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        norm_dic[name] = torch.abs(module.weight.grad).mean(dtype=torch.float32)
    return norm_dic

def get_pre_act(model):
    norm_dic={}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        norm_dic[name] = torch.abs(module.out_data).mean(dtype=torch.float32)
    return norm_dic

def get_pre_act_change(model):
    norm_dic={}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        norm_dic[name] = torch.abs(module.out_data - module.prev_out_data).mean(dtype=torch.float32)
    return norm_dic

def forward_hook(_module, in_data, out_data):
    if not hasattr(_module,'prev_out_data'):
        _module.prev_out_data = out_data.detach().clone()
    _module.out_data = out_data.detach().clone()

def register_fhook(model: torch.nn.Module):
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        module.register_forward_hook(forward_hook)
    return model

def weight_change(model,initial_params):
    total_norm = 0.0
    initial_norm = 0.0
    for current_param, initial_param in zip(model.parameters(), initial_params):
        norm = (current_param - initial_param).pow(2).sum() 
        total_norm += norm.item()
        norm =  (initial_param).pow(2).sum()
        initial_norm  += norm.item()
    return (total_norm / initial_norm)**0.5

# Function to perform word analogy evaluation.
def evaluate_word_analogy(word_embeddings, vocab, analogy_dataset):
    correct_predictions = 0
    all_predictions = 0
    word_to_idx = vocab.get_stoi()
    correct_pairs = []  # List to store correct word pairs
    for analogy in analogy_dataset:
        A, B, C, D = analogy
        if A not in word_to_idx or B not in word_to_idx or C not in word_to_idx or D not in word_to_idx:
            continue
        A_idx = word_to_idx[A]
        B_idx = word_to_idx[B]
        C_idx = word_to_idx[C]
        D_idx = word_to_idx[D]

        # Word analogy calculation: B - A + C
        target_vector = word_embeddings[B_idx] - word_embeddings[A_idx] + word_embeddings[C_idx]

        # Exclude A, B, C from the closest word candidates
        exclude_indices = [A_idx, B_idx, C_idx]
        
        # Calculate cosine similarities
        target_vector_norm = F.normalize(target_vector.unsqueeze(0), p=2, dim=1)
        word_embeddings_norm = F.normalize(word_embeddings, p=2, dim=1)
        similarities = torch.matmul(word_embeddings_norm, target_vector_norm.t()).squeeze()
        
        similarities[exclude_indices] = float('-inf')  # Exclude A, B, C from consideration
        closest_idx = torch.argmax(similarities)

        all_predictions += 1
        if closest_idx == D_idx:
            correct_predictions += 1
            print(f"Correct Answer : {B} - {A} + {C} = {vocab.get_itos()[closest_idx]}")

    accuracy = correct_predictions / all_predictions
    print(f'Analogy Acc : {100*accuracy}% ({correct_predictions} / {all_predictions})')

    return accuracy

def create_analogy_dataset():
    analogy_dataset = []
    with open('./data/analogy/questions-words.txt', 'r') as f:
        lines = f.readlines()

        for line in lines:
            if not line.startswith(':'):
                words = line.strip().split()
                if len(words) == 4:
                    analogy_dataset.append(tuple(words))
                    list1 = [words[2], words[3], words[0], words[1]]
                    list2 = [words[1], words[0], words[3], words[2]]
                    list3 = [words[0], words[1], words[2], words[3]]
                    analogy_dataset.append(tuple(list1))
                    analogy_dataset.append(tuple(list2))
                    analogy_dataset.append(tuple(list3))

    analogy_dataset = (set(analogy_dataset))
    return analogy_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['cbow', 'skipgram'], default='cbow', help='Model name (cbow or skipgram)')
    parser.add_argument('--dataset', type=str, choices=['WikiText2', 'WikiText103'], default='WikiText2', help='Dataset name (WikiText2 or WikiText103)')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=96, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=96, help='Validation batch size')
    parser.add_argument('--shuffle', action='store_false', default=True, help='Whether to shuffle the data')
    parser.add_argument('--model_dir', type=str, default='weights/cbow_WikiText103', help='Path to the directory to save model artifacts')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--train_steps', type=int, default=None, help='Number of training steps per epoch')
    parser.add_argument('--val_steps', type=int, default=None, help='Number of validation steps per epoch')
    parser.add_argument('--optim', type=str, choices=['sgd','adam','adamw'], default='sgd', help='Optimizer name (Adam)')
    parser.add_argument('--lr', type=float, default=0.025, help='Learning rate')
    parser.add_argument('--checkpoint_frequency', type=int, default=1, help='Frequency of saving model checkpoints')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--embed_max_norm', type=int, default=None, help='Embedding max norm')
    parser.add_argument('--min_word_frequency', type=int, default=128, help='min_word_frequency')

    parser.add_argument('--damping', type=float, default=1e-4, help='damping')
    parser.add_argument('--accumulate_iters', type=int, default=-1, help='accumulate_iters')
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum')
    parser.add_argument('--ema_decay', type=float, default=0.1, help='ema_decay')
    parser.add_argument('--curvature_update_interval', type=int, default=10, help='curvature_update_interval')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight_decay')
    parser.add_argument('--loss_type', type=str, default='nll', help='loss_type')

    parser.add_argument('--lr_scheduler', type=str, default='Constant', help='learning rate scheduler')
    parser.add_argument('--damping_technique', type=str, default='martens', help='damping technique')
    parser.add_argument('--exp_gamma', type=float, default=0.9, help='exp_gamma')

    parser.add_argument('--parametrization', type=str, default='SP')
    parser.add_argument('--b_input', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--b_hidden', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--b_output', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--c_input', type=float, default=0,
                        help='learning rate')
    parser.add_argument('--c_hidden', type=float, default=0,
                        help='learning rate')
    parser.add_argument('--c_output', type=float, default=0,
                        help='learning rate')
    parser.add_argument('--input_mult', type=float, default=1,
                        help='input_mult')
    parser.add_argument('--output_mult', type=float, default=1,
                        help='output_mult')
    parser.add_argument('--output_var_mult', type=float, default=1,
                        help='output_mult')
    parser.add_argument('--base_width', type=int, default=128)
    parser.add_argument('--output_zero', action='store_true', default=False)
    parser.add_argument('--embed_std', type=float, default=1)

    parser.add_argument('--bias', action='store_true', default=False)
    parser.add_argument('--log_activation', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_false', default=True)


    args = parser.parse_args()
    if args.embed_max_norm == -1:
        args.embed_max_norm = None
    args.width = args.embed_dim
    config = vars(args).copy()
    if args.wandb:
        wandb.init( config=config,
                    entity=os.environ.get('WANDB_ENTITY', None),
                    project=os.environ.get('WANDB_PROJECT', None),
                    )

    config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "shuffle": args.shuffle,
        "model_dir": args.model_dir,
        "epochs": args.epochs,
        "train_steps": args.train_steps,
        "val_steps": args.val_steps,
        "optimizer": args.optim,
        "learning_rate": args.lr,
        "checkpoint_frequency": args.checkpoint_frequency,
        "embed_dim":args.embed_dim,
        "embed_max_norm":args.embed_max_norm
    }

    #analogy_dataset = [('late', 'early', 'evening', 'morning'), ('man', 'woman', 'king', 'queen'), ('give', 'receive', 'sell', 'buy'), ('female', 'male', 'mother', 'father'), ('hard', 'easy', 'difficult', 'simple'), ('arrive', 'depart', 'enter', 'exit'), ('hot', 'cold', 'summer', 'winter'), ('cat', 'cats', 'dog', 'dogs'), ('rural', 'urban', 'country', 'city'), ('shallow', 'deep', 'low', 'high'), ('early', 'late', 'morning', 'evening'), ('create', 'destroy', 'begin', 'end'), ('school', 'students', 'company', 'employees'), ('shallow', 'pool', 'high', 'mountain'), ('shine', 'sun', 'glow', 'moon'), ('nurse', 'patient', 'teacher', 'student'), ('glow', 'moon', 'shine', 'sun'), ('stream', 'flow', 'street', 'traffic'), ('flow', 'river', 'traffic', 'road'), ('leaf', 'tree', 'petal', 'flower'), ('son', 'daughter', 'brother', 'sister'), ('brother', 'sister', 'uncle', 'aunt'), ('husband', 'wife', 'man', 'woman'), ('rich', 'poor', 'wealthy', 'needy'), ('roof', 'house', 'roof', 'car'), ('teacher', 'student', 'employer', 'employee'), ('best', 'worst', 'good', 'bad'), ('speak', 'speaks', 'write', 'writes'), ('short', 'shorter', 'tall', 'taller'), ('branch', 'tree', 'petal', 'flower'), ('uncle', 'aunt', 'brother', 'sister'), ('read', 'book', 'watch', 'movie'), ('increase', 'decrease', 'expand', 'contract'), ('actress', 'actor', 'director', 'manager'), ('begin', 'end', 'start', 'finish'), ('student', 'teacher', 'employee', 'employer'), ('up', 'down', 'north', 'south'), ('winter', 'summer', 'cold', 'hot'), ('river', 'flow', 'road', 'traffic'), ('population', 'city', 'revenue', 'company'), ('flow', 'river', 'move', 'car'), ('joyful', 'sad', 'positive', 'negative'), ('easy', 'hard', 'simple', 'complex'), ('big', 'small', 'long', 'short'), ('bird', 'birds', 'cat', 'cats'), ('doctor', 'patient', 'teacher', 'student'), ('deep', 'ocean', 'high', 'mountain'), ('city', 'population', 'company', 'revenue'), ('phone', 'call', 'computer', 'program'), ('book', 'read', 'movie', 'watch'), ('sister', 'brother', 'aunt', 'uncle'), ('day', 'night', 'light', 'dark'), ('jump', 'jumped', 'run', 'ran'), ('mother', 'father', 'woman', 'man'), ('actor', 'actress', 'manager', 'director'), ('happy', 'sad', 'positive', 'negative'), ('love', 'hate', 'peace', 'war'), ('king', 'queen', 'man', 'woman'), ('father', 'mother', 'king', 'queen'), ('tall', 'taller', 'small', 'smaller'), ('sun', 'shine', 'moon', 'glow'), ('patient', 'doctor', 'student', 'teacher'), ('mouse', 'computer', 'remote', 'tv'), ('sales', 'company', 'population', 'city'), ('employer', 'employee', 'teacher', 'student'), ('boy', 'girl', 'son', 'daughter'), ('cold', 'hot', 'winter', 'summer'), ('cat', 'purr', 'dog', 'woof'), ('raise', 'lower', 'increase', 'decrease'), ('tree', 'leaf', 'flower', 'petal'), ('daughter', 'son', 'girl', 'boy'), ('walk', 'walked', 'run', 'ran'), ('cat', 'meow', 'dog', 'bark'), ('plant', 'leaf', 'bush', 'branch'), ('old', 'new', 'ancient', 'modern'), ('king', 'queen', 'husband', 'wife'), ('mouse', 'mice', 'louse', 'lice'), ('listen', 'song', 'watch', 'movie'), ('queen', 'king', 'woman', 'man'), ('girl', 'boy', 'daughter', 'son'), ('aunt', 'uncle', 'sister', 'brother'), ('city', 'country', 'urban', 'rural'), ('building', 'roof', 'car', 'hood'), ('large', 'small', 'long', 'short'), ('dog', 'woof', 'cat', 'meow'), ('man', 'woman', 'father', 'mother'), ('ocean', 'deep', 'mountain', 'high'), ('war', 'peace', 'hate', 'love'), ('life', 'death', 'birth', 'funeral'), ('new', 'old', 'modern', 'ancient'), ('nephew', 'niece', 'brother', 'sister'), ('depart', 'arrive', 'exit', 'enter'), ('song', 'listen', 'movie', 'watch'), ('night', 'day', 'dark', 'light'), ('woman', 'man', 'mother', 'father'), ('actress', 'actor', 'director', 'producer'), ('man', 'men', 'child', 'children'), ('manager', 'director', 'actor', 'actress'), ('boy', 'girl', 'brother', 'sister'), ('buy', 'sell', 'lend', 'borrow'), ('woman', 'man', 'queen', 'king'), ('director', 'manager', 'actress', 'actor'), ('mother', 'father', 'wife', 'husband'), ('good', 'better', 'bad', 'worse'), ('rat', 'rats', 'louse', 'lice'), ('moon', 'sun', 'glow', 'shine'), ('queen', 'king', 'wife', 'husband'), ('professor', 'student', 'doctor', 'patient'), ('write', 'writes', 'read', 'reads'), ('father', 'mother', 'man', 'woman'), ('child', 'children', 'person', 'people'), ('producer', 'director', 'employee', 'manager'), ('left', 'right', 'west', 'east'), ('house', 'roof', 'car', 'roof'), ('wealthy', 'poor', 'rich', 'needy'), ('son', 'daughter', 'boy', 'girl'), ('keyboard', 'computer', 'screen', 'phone'), ('student', 'teacher', 'patient', 'doctor'), ('computer', 'keyboard', 'phone', 'screen'), ('hood', 'car', 'roof', 'house'), ('death', 'life', 'funeral', 'birth'), ('dog', 'bark', 'cat', 'meow'), ('leader', 'follower', 'manager', 'employee'), ('teacher', 'student', 'doctor', 'patient')]
    analogy_dataset = create_analogy_dataset()

    if args.parametrization == 'SP':
        args.b_output=1/2
        args.b_input=1/2
        args.b_hidden=1/2
        args.c_output=0
        args.c_input=0
        args.c_hidden=0
    if args.parametrization == 'SP_LR':
        if args.optim == 'sgd':
            args.b_output=1/2
            args.b_input=1/2
            args.b_hidden=1/2
            args.c_output=1
            args.c_input=1
            args.c_hidden=1
        if 'kfac' in args.optim:
            args.b_output=1/2
            args.b_input=1/2
            args.b_hidden=1/2
            args.c_output=0
            args.c_input=0
            args.c_hidden=0
    if args.parametrization == 'UP':
        if args.optim == 'sgd':
            args.b_output=1/2
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1
            args.c_hidden=1
            args.c_input=0
        if 'kfac' in args.optim:
            args.b_output=1/2
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=0
            args.c_input=0
        if args.optim == 'shampoo':
            args.b_output=1/2
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1/2
            args.c_hidden=1/2
            args.c_input=0
        if 'foof' in args.optim:
            args.b_output=1/2
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=0
            args.c_input=0
    if args.parametrization == 'muP':
        if args.optim == 'sgd':
            args.b_output=1
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1
            args.c_hidden=0
            args.c_input=-1
        if 'kfac' in args.optim:
            args.b_output=1
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=0
            args.c_input=0
        if args.optim == 'shampoo':
            args.b_output=1
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1/2
            args.c_hidden=0
            args.c_input=-1/2
        if 'foof' in args.optim:
            args.b_output=1
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=-1
            args.c_input=-1
    if args.parametrization == 'muP_output_zero':
        args.output_zero = True
        if args.optim == 'sgd':
            args.b_output=128
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1
            args.c_hidden=0
            args.c_input=-1
        if 'kfac' in args.optim:
            args.b_output=128
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=0
            args.c_input=0
        if args.optim == 'shampoo':
            args.b_output=128
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1/2
            args.c_hidden=0
            args.c_input=-1/2
        if 'foof' in args.optim:
            args.b_output=128
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=-1
            args.c_input=-1
    if args.parametrization == 'class_muP':
        if args.optim == 'sgd':
            args.b_output=1/2
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=0
            args.c_input=-1
    if args.parametrization == 'class_muP_output_zero':
        args.output_zero = False
        if args.optim == 'sgd':
            args.b_output=128
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=0
            args.c_hidden=0
            args.c_input=-1
    if args.parametrization == 'kfac_muP':
        args.b_output=1
        args.b_hidden=1/2
        args.b_input=1/2
        args.c_output=0
        args.c_hidden=0
        args.c_input=0

    train()