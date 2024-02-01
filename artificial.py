import sys,os
sys.path.append(os.path.abspath('../../'))
sys.path.append('./utils/')

import argparse
import numpy as np
import time
import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os,json
import utils.dataset
from models.create_model import create_model,initialize_weight, create_finetune_model
from utils.damping import set_damping
import wandb
from timm.scheduler import CosineLRScheduler
from utils.loss_type import CustomCrossEntropyLoss, CustomMSELossWithL2Reg
from utils.create_optim import create_optimizer, create_optimizer_for_head
import warmup_scheduler
import random
from utils.custom_dataset import CustomDataset

dataset_options = ['MNIST','CIFAR10','CIFAR100','SVHN','Flowers','Cars', 'FashionMNIST', 'STL10']

max_validation_acc=0
min_validation_loss=np.inf
max_train_acc=0
min_train_loss=np.inf
max_train_acc_all=0
min_train_loss_all=np.inf

os.environ["WANDB_HOST"] = os.environ.get('SLURM_JOBID')

def main(epochs,  prefix = ''):
    total_train_time=0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train(epoch, prefix)
        total_train_time += time.time() - start
        trainloss_all(epoch, prefix)
        if args.log_h_delta:
            log_h_delta(epoch, prefix)
    print(f'total_train_time: {total_train_time:.2f}s')
    print(f'avg_epoch_time: {total_train_time / args.epochs:.2f}s')
    if args.wandb:
        wandb.run.summary['total_train_time'] = total_train_time
        wandb.run.summary['avg_epoch_time'] = total_train_time / args.epochs

def trainloss_all(epoch, prefix = ''):
    global max_train_acc_all,min_train_loss_all
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    
    if train_accuracy>max_train_acc_all:
        max_train_acc_all=train_accuracy
    if train_loss<min_train_loss_all:
        min_train_loss_all=train_loss

    if args.wandb:
        log = {prefix + 'epoch': epoch,
               prefix + 'iteration': (epoch) * len(train_loader.dataset),
               prefix + 'train_loss_all': train_loss,
               prefix + 'train_accuracy_all': train_accuracy,
               prefix + 'max_train_acc_all':max_train_acc_all,
               prefix + 'min_train_loss_all':min_train_loss_all}
        wandb.log(log)
    print('Train all set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset), train_accuracy))

    if math.isnan(train_loss):
        print('Error: Train loss is nan', file=sys.stderr)
        return True
    return False

def train(epoch, prefix = ''):
    global max_train_acc,min_train_loss
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (x, t) in enumerate(train_loader):
        model.train()
        x, t = x.to(device), t.to(device)
        
        if args.loss_type == 'cross_entropy':
            if args.noise_eps>0 or args.class_reduction:
                loss_func = CustomCrossEntropyLoss(epsilon = args.noise_eps, label_smoothing=args.label_smoothing, reduction=args.class_reduction_type)
                t2 = t
            else:
                loss_func = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
                t2 = t
        elif args.loss_type=='mse':
            if args.noise_eps>0 or args.class_reduction:
                loss_func = CustomMSELossWithL2Reg(model=model, lambda_reg=0, reduction=args.class_reduction_type)
            else:
                loss_func = torch.nn.MSELoss()
            t2 = MSE_label(x, t)

        y = model(x)
        loss = loss_func(y,t2)
        loss.backward()
        grad_norm = get_grad_norm(model)

        if args.loss_type == 'cross_entropy':
            t3 = t2
            if t2.ndim == 1:
                t3 = F.one_hot(t3, num_classes=y.size(1)).float()
            chi_norm = torch.abs(F.log_softmax(y, dim=1)-t3).mean(dtype=torch.float32).item()
        elif args.loss_type=='mse':
            chi_norm = torch.abs(y-t2).mean(dtype=torch.float32).item()

        if args.chi_fixed:
            current_lr = optimizer.param_groups[0]['lr']  # 現在の学習率を取得
            adjust_learning_rate(optimizer, current_lr/chi_norm)  # 学習率を更新
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if args.chi_fixed:
            adjust_learning_rate(optimizer, current_lr)  # 学習率を更新
        
        if batch_idx%1==0:
            pred = y.data.max(1)[1]
            acc = 100. * pred.eq(t.data).cpu().sum() / t.size(0)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx* len(x) / len(train_loader.dataset), float(loss), float(acc)))
            l1_norm, l2_norm = get_weight_norm(model)
            if acc>max_train_acc:
                max_train_acc=acc
            if loss<min_train_loss:
                min_train_loss=loss
            log = {prefix + 'epoch': epoch,
                   prefix + 'iteration': batch_idx + (epoch-1) * len(train_loader.dataset),
                   prefix + 'train_loss': float(loss),
                   prefix + 'train_accuracy': float(acc),
                   prefix + 'max_train_acc':max_train_acc,
                   prefix + 'min_train_loss':min_train_loss,
                   prefix + 'l1_norm':l1_norm,
                   prefix + 'l2_norm':l2_norm,
                   prefix + 'grad_norm_all':grad_norm,
                   prefix + 'chi_norm':chi_norm,
                   prefix + 'chi_norm_inv':1/chi_norm,
                   }
            if args.wandb:
                wandb.log(log)

    if scheduler is not None:
        scheduler.step(epoch=epoch)

# 学習率を調整する関数
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_weight_norm(model):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return l1_norm.item(), (l2_norm.item()**0.5)

def get_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

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

def MSE_label(output, target):
    y_onehot = output.new_zeros(output.size(0), dataset.num_classes)
    y_onehot.scatter_(1, target.unsqueeze(-1), 1)
    if not args.spaese_coding_mse:
        y_onehot -= 1/dataset.num_classes
    return y_onehot

def register_fhook(model: torch.nn.Module):
    # forward hook function
    def forward_hook(_module, in_data, out_data):
      _module.out_data = out_data.detach().clone()
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        module.register_forward_hook(forward_hook)
    return model

def fetch_h(model):
    model = register_fhook(model)
    y = model(inputs_for_dh)
    pre_act_dict = {}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
              continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        pre_act_dict[name] = module.out_data
    return pre_act_dict

def log_h_delta(epoch, prefix = ''):
    pre_act_dict = fetch_h(model)
    h_norm_dict = {}
    dh_norm_dict = {}
    for mname in pre_act_dict.keys():
        h_norm_dict[mname] = torch.abs(pre_act_dict[mname]).mean(dtype=torch.float32).item()
        dh_norm_dict[mname] = torch.abs(pre_act_dict[mname] - init_pre_act_dict[mname]).mean(dtype=torch.float32).item()
    print(dh_norm_dict)
    if args.wandb:
        log = {prefix + 'epoch': epoch,
               prefix + 'iteration': epoch * len(train_loader.dataset),
               prefix + 'h/':h_norm_dict,
               prefix + 'dh/': dh_norm_dict,}
        wandb.log(log)

class ParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)

def muP_set(args):
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
        args.output_nonzero = False
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
        args.output_nonzero = False
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='MNIST',
                        choices=dataset_options)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--base_width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--bias', action='store_true', default=False)

    parser.add_argument('--output_mult', type=float, default=1)
    parser.add_argument('--input_mult', type=float, default=1)
    parser.add_argument('--init_std', type=float, default=1)

    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pseudo_batch_size', type=int, default=-1,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--head_init_epochs', type=int, default=-1,
                        help='number of iterations to train head (default: 0)')
    parser.add_argument('--head_init_iterations', type=int, default=-1,
                        help='number of iterations to train head (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    
    parser.add_argument('--label_smoothing', type=float, default=0,
                        help='label_smoothing')

    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=None,
                        help='length of the holes')
    parser.add_argument('--RandomCrop', action='store_true', default=False)
    parser.add_argument('--RandomHorizontalFlip', action='store_true', default=False)
    parser.add_argument('--CIFAR10Policy', action='store_true', default=False)
    parser.add_argument('--dataset_shuffle', action='store_true', default=False)

    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='learning rate')
    parser.add_argument('--init_lr', type=float, default=3e-3,
                        help='learning rate')
    parser.add_argument('--init_momentum', type=float, default=0,
                        help='learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lambda_reg', type=float, default=0)
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--load_base_shapes', type=str, default='width64.bsh',
                        help='file location to load base shapes from')

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
    parser.add_argument('--output_var_mult', type=float, default=1,
                        help='learning rate')

    parser.add_argument('--activation', type=str, default='relu',
                        help='act')
    parser.add_argument('--norm_type', type=str, default='l1',
                        help='log_type')
    
    parser.add_argument('--parametrization', type=str, default='SP')
    parser.add_argument('--output_nonzero', action='store_true', default=False)
    parser.add_argument('--curvature_update_interval', type=int, default=1)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--sched_power', type=float, default=1,
                        help='sched_power')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='sched_power')
    
    parser.add_argument('--save_ckpt', action='store_true', default=False)

    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_val_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_weight_delta', action='store_true', default=False,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_h_delta', action='store_true', default=False,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_damping', action='store_true', default=False,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--train_size', type=int, default=1024)

    parser.add_argument('--widen_factor', type=int, default=4)
    parser.add_argument('--num_features', type=int, default=4)

    parser.add_argument('--imbalance_alpha', type=float, default=None)

    parser.add_argument('--ckpt_path', type=str, default='./ckpt/mlp/')
    parser.add_argument('--ckpt_interval', type=int, default=10)

    parser.add_argument('--loss_type', type=str, default='cross_entropy')

    parser.add_argument('--log_record', action='store_true', default=False)
    parser.add_argument('--use_timm', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--wandb', action='store_false', default=True)

    parser.add_argument('--withoutShortcut', action='store_true', default=False)
    parser.add_argument('--withoutBN', action='store_true', default=False)
    parser.add_argument('--finetuning', action='store_true', default=False)
    parser.add_argument('--noise_eps', type=float, default=0)
    parser.add_argument('--class_reduction', action='store_true', default=False)
    parser.add_argument('--class_reduction_type', type=str, default='mean')

    parser.add_argument('--chi_fixed', action='store_true', default=False)
    parser.add_argument('--spaese_coding_mse', action='store_true', default=False)

    parser.add_argument('--config', default=None,
                        help='config file path')

    args = parser.parse_args()
    dict_args = vars(args)
    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        dict_args.update(config)
    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.job_id = os.environ.get('SLURM_JOBID')
    cudnn.benchmark = True  # Should make training should go faster for large models
    config = vars(args).copy()

    if args.wandb:
        wandb.init( config=config,
                    entity=os.environ.get('WANDB_ENTITY', None),
                    project=os.environ.get('WANDB_PROJECT', None),
                    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')
    
    num_classes = args.width
    dataset = CustomDataset(num_samples=args.train_size, 
                            num_features=args.num_features*args.num_features, 
                            num_classes=num_classes, 
                            imbalance_alpha = args.imbalance_alpha)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.pseudo_batch_size != -1:
        args.accumulate_iters = args.pseudo_batch_size / args.batch_size
    else:
        args.accumulate_iters=1
    muP_set(args)
    
    model = create_model(args.num_features, dataset.num_classes, dataset.num_channels, args).to(device=device)
    model = initialize_weight(model,b_input=args.b_input,b_hidden=args.b_hidden,b_output=args.b_output,output_nonzero=args.output_nonzero,output_var_mult=args.output_var_mult)

    print(model)

    if args.log_weight_delta:
        initial_params = [param.clone() for param in model.parameters()]
    optimizer = create_optimizer(args, model, lr = args.lr)
    scheduler=None
    if args.scheduler == 'CosineAnnealingLR':
        scheduler=CosineLRScheduler(optimizer, t_initial=args.epochs,lr_min=0, warmup_t=args.warmup_epochs)

    if args.log_h_delta:
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs_for_dh = inputs.to(device)
            break
        init_pre_act_dict = fetch_h(model)
    try:
        main(epochs=args.epochs, prefix='')
    except ValueError as e:
        print(e)
    
    wandb.finish()