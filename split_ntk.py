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
from torch.func import vmap, jacrev

import os,json
import utils.dataset
from models.create_model import create_model,initialize_weight, MultiHeadModel
from utils.damping import set_damping
import wandb
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR, PolynomialLR
from utils.loss_type import CustomCrossEntropyLoss, CustomMSELossWithL2Reg
from utils.create_optim import create_optimizer, create_optimizer_for_head, create_spectral_optimizer
import warmup_scheduler
import random

dataset_options = ['MNIST','CIFAR10','CIFAR100','SVHN','Flowers','Cars', 'FashionMNIST', 'STL10']

max_validation_acc=0
min_validation_loss=np.inf
max_train_acc=0
min_train_loss=np.inf
max_train_acc_all=0
min_train_loss_all=np.inf

job_id = os.environ.get('SLURM_JOBID')
if job_id is not None:
    os.environ["WANDB_HOST"] = job_id


class ParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)

def muP_set(args):
    if args.parametrization == 'SP':
        args.output_nonzero = True
        args.b_output=1/2
        args.b_input=1/2
        args.b_hidden=1/2
        args.c_output=0
        args.c_input=0
        args.c_hidden=0
    if args.parametrization == 'SP_output_zero':
        args.output_nonzero = False
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
        args.output_nonzero = True
        if args.optim == 'sgd':
            args.b_output=1
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1
            args.c_hidden=0
            args.c_input=-1
        if args.optim == 'adam':
            args.b_output=1
            args.b_hidden=1/2
            args.b_input=1/2
            args.c_output=1
            args.c_hidden=1
            args.c_input=0
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
    if args.parametrization == 'Spectral_output_zero':
        args.output_nonzero = False

def fnet_single(params, x):
    x = x.view(-1, 32*32*3)  # 32x32x3に変更
    x = F.linear(x, params[0], None)
    x = F.layer_norm(x, [1024], params[2], params[3])
    x = F.relu(x)
    x = F.linear(x, params[1], None)
    x = F.layer_norm(x, [1024], params[2], params[3])
    x = F.relu(x)
    x = F.linear(x, params[6], params[7])
    return x

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, block_size=32):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1_flat = torch.cat([j.reshape(j.shape[0], -1) for j in jac1], dim=1)
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2_flat = torch.cat([j.reshape(j.shape[0], -1) for j in jac2], dim=1)

    # Initialize result matrix
    num_x1 = jac1_flat.shape[0]
    num_x2 = jac2_flat.shape[0]
    result = torch.zeros((num_x1, num_x2), device=jac1_flat.device)

    # Block-wise computation to save memory
    for i in range(0, num_x1, block_size):
        for j in range(0, num_x2, block_size):
            block_jac1 = jac1_flat[i:i+block_size]
            block_jac2 = jac2_flat[j:j+block_size]
            result[i:i+block_size, j:j+block_size] = block_jac1 @ block_jac2.T
    return result

def predict_with_ntk(ntk, x_train, y_train, x_test, reg=1e-4):
    # Compute Gram matrix and its inverse
    K_train_train = ntk[:len(x_train), :len(x_train)]
    K_train_train += reg * torch.eye(K_train_train.shape[-1], device=device)  # Regularization
    K_train_train_inv = torch.inverse(K_train_train)
    alpha = K_train_train_inv @ (y_train - model(x_train, task=1)).float()

    preds_list = []
    for i in range(0, len(x_test), 128):
        x_test_batch = x_test[i:i+128]
        K_train_test = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test_batch)
        preds_batch = (model(x_test_batch, task=1) + K_train_test.transpose(0, 1) @ alpha).argmax(dim=1)
        preds_list.append(preds_batch)
    
    return torch.cat(preds_list, dim=0)

def predict_with_ntk(ntk, x_train, y_train, x_test, reg=1e-4):
    # Compute Gram matrix and its inverse
    K_train_train = ntk[:len(x_train), :len(x_train)]
    K_train_train += reg * torch.eye(K_train_train.shape[-1], device=device)  # Regularization
    K_train_train_inv = torch.inverse(K_train_train)
    alpha = K_train_train_inv @ (y_train - model(x_train, task=1)).float()

    preds_list = []
    for i in range(0, len(x_test), 128):
        x_test_batch = x_test[i:i+128]
        K_train_test = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test_batch)
        preds_batch = (model(x_test_batch, task=1) + K_train_test.transpose(0, 1) @ alpha).argmax(dim=1)
        preds_list.append(preds_batch)
    
    return torch.cat(preds_list, dim=0)

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

    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pseudo_batch_size', type=int, default=-1,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--train_acc_stop', type=float, default=None,
                        help='train_acc_stop (default: 20)')
    parser.add_argument('--pretrained_epochs', type=int, default=100,
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

    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--pretrained_lr', type=float, default=1e-2,
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
    parser.add_argument('--ckpt_folder', type=str, default='./lp_ckpts/mlp_ln_split_cifar_aug_10_8_2/')

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
    parser.add_argument('--dA', type=float, default=-1,
                        help='learning rate')
    parser.add_argument('--dB', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--output_var_mult', type=float, default=1,
                        help='learning rate')

    parser.add_argument('--activation', type=str, default='relu',
                        help='act')
    parser.add_argument('--norm_type', type=str, default='l1',
                        help='log_type')
    parser.add_argument('--multihead', action='store_true', default=False)
    
    parser.add_argument('--parametrization', type=str, default='SP')
    parser.add_argument('--pretrained_parametrization', type=str, default='muP_output_zero')

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
    parser.add_argument('--log_dh_interval', type=int, default=2,
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
    parser.add_argument('--train_size', type=int, default=128)
    parser.add_argument('--pretrained_train_size', type=int, default=-1)

    parser.add_argument('--widen_factor', type=int, default=4)
    parser.add_argument('--loss_type', type=str, default='mse')

    parser.add_argument('--log_record', action='store_true', default=False)
    parser.add_argument('--use_timm', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--population_coding', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_false', default=True)

    parser.add_argument('--withoutShortcut', action='store_true', default=False)
    parser.add_argument('--withoutBN', action='store_true', default=False)
    parser.add_argument('--class_scaling', action='store_true', default=False)
    parser.add_argument('--real_class_scaling', action='store_true', default=False)
    parser.add_argument('--class_bulk', action='store_true', default=False)
    parser.add_argument('--finetuning', action='store_true', default=False)
    parser.add_argument('--noise_eps', type=float, default=0)
    parser.add_argument('--class_reduction', action='store_true', default=False)
    parser.add_argument('--class_reduction_type', type=str, default='mean')
    parser.add_argument('--permutate', action='store_true', default=False)
    parser.add_argument('--train_classes', type=str, default=None)
    parser.add_argument('--pretrained_classes', type=str, default=None)
    parser.add_argument('--task1_class', type=int, default=10)
    parser.add_argument('--task2_class', type=int, default=10)

    parser.add_argument('--chi_fixed', action='store_true', default=False)
    parser.add_argument('--spaese_coding_mse', action='store_true', default=False)
    parser.add_argument('--RandomAffine', action='store_true', default=False)
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

    if args.train_classes is not None:
        args.train_classes = args.train_classes.split(',')
        args.train_classes = [int(c) for c in args.train_classes]
    if args.pretrained_classes is not None:
        args.pretrained_classes = args.pretrained_classes.split(',')
        args.pretrained_classes = [int(c) for c in args.pretrained_classes]

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    job_id = os.environ.get('SLURM_JOBID')
    if job_id is not None:
        args.job_id = job_id
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
    
    if args.real_class_scaling:
        if args.dataset == 'CIFAR100':
            num_classes = int(6 * args.width / args.base_width)
    else:
        num_classes = -1

    if args.dataset == 'MNIST':
        dataset = utils.dataset.MNIST(args=args, permutate=args.permutate)
    elif args.dataset == 'FashionMNIST':
        dataset = utils.dataset.FashionMNIST(args=args)
    elif args.dataset == 'CIFAR10':
        dataset = utils.dataset.CIFAR10(args=args, task_classes=args.train_classes)

    dataset_original_class = dataset.num_classes
    if args.class_scaling or args.population_coding:
        dataset.num_classes *= int(args.width / args.base_width)
    args.task1_class_head = args.task1_class
    args.task2_class_head = args.task2_class
    if args.population_coding:
        args.task1_class = dataset.num_classes
        args.task2_class = dataset.num_classes
    print("args.population_coding", args.population_coding, "args.task1_class, args.task2_class", args.task1_class, args.task2_class)

    if args.pseudo_batch_size != -1:
        args.accumulate_iters = args.pseudo_batch_size / args.batch_size
    else:
        args.accumulate_iters=1
        
    muP_set(args)

    if args.head_init_epochs == -1:
        if args.head_init_iterations != -1:
            args.head_init_epochs = 1 + args.head_init_iterations // dataset.num_steps_per_epoch
        elif args.head_init_iterations == -1:
            args.head_init_epochs = 0

    if args.pseudo_batch_size != -1:
        args.batch_size=args.pseudo_batch_size
    
    if args.multihead:
        args.task1_parametrization = args.parametrization
        args.task2_parametrization = args.parametrization
        args.use_cifar_model = True
        model = MultiHeadModel(args, dataset.img_size, dataset.num_classes, dataset.num_channels).to(device=device)
    else:
        model = create_model(dataset.img_size, dataset.num_classes, dataset.num_channels, args).to(device=device)
        model = initialize_weight(model,b_input=args.b_input,b_hidden=args.b_hidden,b_output=args.b_output,output_nonzero=args.output_nonzero,output_var_mult=args.output_var_mult)

    if args.pretrained_epochs > 0:
        # linear probingをtsizeで行ったfile name
        file_name = str(args.model) + '_' + str(args.dataset)  + '_wid_' + str(args.width) + '_ep_' + str(args.pretrained_epochs) + '_hp_' + str(args.head_init_epochs) +'_pretrained_param_' + str(args.pretrained_parametrization)+'_param_' + str(args.parametrization) + '_tsize_' + str(args.train_size) + '_lr_' + str(args.pretrained_lr) + '_loss_' + str(args.loss_type) + '_act_' + str(args.activation) + '.pt'
        folder_path = os.path.join(args.ckpt_folder, file_name)
        checkpoint = torch.load(folder_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.population_coding:
            pretrained_orthogonal_matrix = checkpoint['pretrained_orthogonal_matrix'][:args.task1_class_head, :]
            orthogonal_matrix = checkpoint['orthogonal_matrix'][:args.task2_class_head, :]

    params = [p for p in model.parameters()]
    for idx, (name, module) in enumerate(model.named_parameters()):
        print(idx, name, module.size())
    x_train, y_train = next(iter(dataset.train_loader))
    # デバイスに転送
    x_train, y_train = x_train.to(device), y_train.to(device)
    ntk = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_train)
    y_train_one_hot = F.one_hot(y_train-8, num_classes=2).float()

    # テストデータをバッチごとに処理
    test_preds_list = []
    test_targets_list = []
    with torch.no_grad():
        for x_batch, y_batch in dataset.val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)-8
            preds = predict_with_ntk(ntk, x_train, y_train_one_hot, x_batch)
            test_preds_list.append(preds)
            test_targets_list.append(y_batch)

    test_preds = torch.cat(test_preds_list, dim=0)
    test_targets = torch.cat(test_targets_list, dim=0)

    # 精度の計算
    accuracy = (test_preds == test_targets).float().mean().item()
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    if args.wandb:
        wandb.log({
            'val_accuracy' : accuracy * 100
        })
    
    wandb.finish()