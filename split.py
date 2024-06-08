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
from models.create_model import create_model,initialize_weight, MultiHeadModel
import wandb
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR, PolynomialLR
from utils.loss_type import CustomCrossEntropyLoss, CustomMSELossWithL2Reg
from utils.create_optim import create_optimizer, create_optimizer_for_head, create_spectral_optimizer
import warmup_scheduler
from functorch import make_functional_with_buffers
from asdl.kernel import empirical_class_wise_direct_ntk

import copy
from models.linear_model import LinearizedModel
from utils.set_mup import muP_set

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

def main(epochs, iterations = -1, prefix = '', linear_training = False):
    total_train_time=0

    # First Acc
    trainloss_all(0, pretrained_dataset, prefix+'pretrained_')
    val(0, pretrained_dataset, prefix+'pretrained_')
    trainloss_all(0, dataset, prefix, multihead=args.multihead)
    val(0, dataset, prefix, multihead=args.multihead)
    wandb.run.summary["first_val_accuracy"] = max_validation_acc

    for epoch in range(1, epochs + 1):
        start = time.time()
        train(epoch, prefix, iterations, multihead=args.multihead, linear_training=linear_training)
        total_train_time += time.time() - start
        trainloss_all(epoch, pretrained_dataset, prefix+'pretrained_')
        val(epoch, pretrained_dataset, prefix+'pretrained_')
        train_accuracy = trainloss_all(epoch, dataset, prefix, multihead=args.multihead)
        nantf = val(epoch, dataset, prefix, multihead=args.multihead)
        if args.log_h_delta:
            log_h_delta(epoch, prefix)
        if nantf:
            break
        if args.train_acc_stop is not None and train_accuracy > args.train_acc_stop:
            wandb.run.summary['total_epochs_task'] = epoch
            break
        delta_ntk(epoch, model, initial_model, dataset, multihead=args.multihead)
    print(f'total_train_time: {total_train_time:.2f}s')
    print(f'avg_epoch_time: {total_train_time / args.epochs:.2f}s')
    print(f'avg_step_time: {total_train_time / args.epochs / dataset.num_steps_per_epoch * 1000:.2f}ms')
    if args.wandb:
        wandb.run.summary['total_train_time'] = total_train_time
        wandb.run.summary['avg_epoch_time'] = total_train_time / args.epochs
        wandb.run.summary['avg_step_time'] = total_train_time / args.epochs / dataset.num_steps_per_epoch

def val(epoch, dataset, prefix = '', multihead=False):
    global max_validation_acc,min_validation_loss
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset.val_loader:
            data, target = data.to(device), target.to(device)
            if 'pretrained' not in prefix:
                target -= args.task1_class_head
            if multihead and 'pretrained' not in prefix:
                output = model(data, task=1)
            else:
                output = model(data)
            if args.population_coding:
                if 'pretrained' in prefix:
                    target2 = pretrained_orthogonal_matrix[target]
                else:
                    target2 = orthogonal_matrix[target]
                test_loss += F.mse_loss(output, target2, reduction='sum').item()
            else:
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
            if args.population_coding:
                if 'pretrained' in prefix:
                    pred = (output@pretrained_orthogonal_matrix.T).argmax(dim=1, keepdim=True)
                else:
                    pred = (output@orthogonal_matrix.T).argmax(dim=1, keepdim=True)
            else:
                pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataset.val_loader.dataset)
    test_accuracy = 100. * correct / len(dataset.val_loader.dataset)
    if test_accuracy>max_validation_acc:
        max_validation_acc=test_accuracy
    if test_loss<min_validation_loss:
        min_validation_loss=test_loss

    if args.wandb:
        log = {prefix + 'epoch': epoch,
               prefix + 'iteration': epoch * dataset.num_steps_per_epoch,
               prefix + 'val_loss': test_loss,
               prefix + 'val_accuracy': test_accuracy,
               prefix + 'max_val_accuracy':max_validation_acc,
               prefix + 'min_val_loss':min_validation_loss}
        wandb.log(log)
    print('Epoch {:.0f} = Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, len(dataset.val_loader.dataset), test_accuracy))

    if math.isnan(test_loss):
        print('Error: Train loss is nan', file=sys.stderr)
        return True
    return False

def trainloss_all(epoch, dataset, prefix = '', multihead=False):
    global max_train_acc_all,min_train_loss_all
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset.train_val_loader:
            data, target = data.to(device), target.to(device)
            if 'pretrained' not in prefix:
                target -= args.task1_class_head
            if multihead and 'pretrained' not in prefix:
                output = model(data, task=1)
            else:
                output = model(data)
            if args.population_coding:
                if 'pretrained' in prefix:
                    target2 = pretrained_orthogonal_matrix[target]
                else:
                    target2 = orthogonal_matrix[target]
                train_loss += F.mse_loss(output, target2).item()
            else:
                train_loss += F.cross_entropy(output, target).item()
            if args.population_coding:
                if 'pretrained' in prefix:
                    pred = (output@pretrained_orthogonal_matrix.T).argmax(dim=1, keepdim=True)
                else:
                    pred = (output@orthogonal_matrix.T).argmax(dim=1, keepdim=True)
            else:
                pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(dataset.train_val_loader.dataset)
    train_accuracy = 100. * correct / len(dataset.train_val_loader.dataset)
    
    if train_accuracy>max_train_acc_all:
        max_train_acc_all=train_accuracy
    if train_loss<min_train_loss_all:
        min_train_loss_all=train_loss

    if args.wandb:
        log = {prefix + 'epoch': epoch,
               prefix + 'iteration': (epoch) * dataset.num_steps_per_epoch,
               prefix + 'train_loss_all': train_loss,
               prefix + 'train_accuracy_all': train_accuracy,
               prefix + 'max_train_accuracy_all':max_train_acc_all,
               prefix + 'min_train_loss_all':min_train_loss_all}
        wandb.log(log)
    print('Epoch {:.0f} = Train all set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, train_loss, correct, len(dataset.val_loader.dataset), train_accuracy))

    if math.isnan(train_loss):
        print('Error: Train loss is nan', file=sys.stderr)
        return True
    return train_accuracy

def train(epoch, prefix = '', train_iterations=-1, multihead=False, linear_training = False):
    global max_train_acc,min_train_loss
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (x, t) in enumerate(dataset.train_loader):
        if train_iterations != -1 and (epoch-1) * dataset.num_steps_per_epoch+batch_idx >= train_iterations:
            return
        model.train()
        x, t = x.to(device), t.to(device)
        t -= args.task1_class_head

        if args.population_coding:
            loss_func = torch.nn.MSELoss()
            t2 = orthogonal_matrix[t]
        elif args.loss_type == 'cross_entropy':
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

        if linear_training:
            if multihead:
                y = initial_model(x, task=1)
            else:
                y = initial_model(x)
        else:
            if multihead:
                y = model(x, task=1)
            else:
                y = model(x)
        loss = loss_func(y,t2)
        loss.backward()
        if linear_training:
            for (name_a, param_init), (name_b, param_model) in zip(initial_model.named_parameters(), model.named_parameters()):
                if param_init.grad is not None:
                    param_model.grad = param_init.grad.clone()

        if batch_idx%args.accumulate_iters == args.accumulate_iters-1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            initial_model.zero_grad() 

        if batch_idx%100==0 and args.wandb:
            if multihead:
                y = model(x, task=1)
            else:
                y = model(x)
            loss = loss_func(y,t2)
            if args.population_coding:
                pred = (y@orthogonal_matrix.T).data.max(1)[1]
            else:
                pred = y.data.max(1)[1]
            acc = 100. * pred.eq(t.data).cpu().sum() / t.size(0)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(dataset.train_loader.dataset),
                100. * batch_idx / dataset.num_steps_per_epoch, float(loss)))
            with torch.no_grad():
                init_tensor = get_model_parameters_tensor(initial_model)
                mtensor = get_model_parameters_tensor(model)
                norm_layer_dic, abs_norm_layer_dic = get_weight_norm_delta(model, initial_model, spectral=False)
                
            if acc>max_train_acc:
                max_train_acc=acc
            if loss<min_train_loss:
                min_train_loss=loss
            log = {prefix + 'epoch': epoch,
                   prefix + 'iteration': (epoch-1) * dataset.num_steps_per_epoch+batch_idx,
                   prefix + 'train_loss': float(loss),
                   prefix + 'train_accuracy': float(acc),
                   prefix + 'max_train_accuracy':max_train_acc,
                   prefix + 'min_train_loss':min_train_loss,
                   prefix + 'dif/l2_':torch.norm(mtensor - init_tensor) / torch.norm(mtensor),
                   prefix + 'dif/abs_':torch.abs(mtensor - init_tensor).mean(dtype=torch.float32).item() / torch.abs(mtensor).mean(dtype=torch.float32).item(),
                   prefix + 'layer_dif/l2_':norm_layer_dic,
                   prefix + 'layer_dif/abs_':abs_norm_layer_dic
                   }
            wandb.log(log)

    if scheduler is not None:
        scheduler.step(epoch=epoch)

def get_model_parameters_tensor(model):
    if hasattr(model, 'params'):
        params = model.params
    else:
        _, params, _ = make_functional_with_buffers(
                model, disable_autograd_tracking=True
            )
        params = torch.nn.ParameterList(params)
    parameters = []
    for param in params:
        parameters.append(param.view(-1))  # Flatten each parameter and add to the list
    return torch.cat(parameters)

def get_weight_norm_delta(model_a, model_b, spectral=True):
    norm_layer_dic={}
    abs_norm_layer_dic={}
    spectral_norm_layer_dic = {}
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if name_a != name_b:
            raise ValueError(f"Parameter names do not match: {name_a} vs {name_b}")
        diff = param_a - param_b
        norm_layer_dic[name_a] = torch.norm(diff, p=2) / torch.norm(param_a, p=2)
        abs_norm_layer_dic[name_a] = torch.abs(diff).mean(dtype=torch.float32).item() / torch.abs(param_a).mean(dtype=torch.float32).item()
        if spectral:
            spectral_norm_layer_dic[name_a] = torch.linalg.norm(diff,ord=2) / torch.linalg.norm(param_a,ord=2)
    if spectral:
        return norm_layer_dic, abs_norm_layer_dic, spectral_norm_layer_dic
    else:
        return norm_layer_dic, abs_norm_layer_dic

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
    if args.multihead:
        dataset_num_classes = args.task2_class
    else:
        dataset_num_classes = dataset.num_classes
    y_onehot = output.new_zeros(output.size(0), dataset_num_classes)
    y_onehot.scatter_(1, target.unsqueeze(-1), 1)
    if not args.spaese_coding_mse:
        y_onehot -= 1/dataset_num_classes
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

def fetch_h(model, multihead=False):
    model = register_fhook(model)
    if multihead:
        y = model(inputs_for_dh, task=1)
    else:
        y = model(inputs_for_dh)
    pre_act_dict = {}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
              continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        if hasattr(module, 'out_data'):
            pre_act_dict[name] = module.out_data
    return pre_act_dict

def log_h_delta(epoch, prefix = ''):
    global tmp_pre_act_dict
    pre_act_dict = fetch_h(model, args.multihead)
    h_norm_dict = {}
    dh_norm_dict = {}
    int_dh_norm_dict = {}
    for mname in pre_act_dict.keys():
        if 'head' in mname:
            continue
        h_norm_dict[mname] = torch.abs(pre_act_dict[mname]).mean(dtype=torch.float32).item()
        dh_norm_dict[mname] = torch.abs(pre_act_dict[mname] - init_pre_act_dict[mname]).mean(dtype=torch.float32).item()
        int_dh_norm_dict[mname] = torch.abs(pre_act_dict[mname] - tmp_pre_act_dict[mname]).mean(dtype=torch.float32).item()
    log = {prefix + 'epoch': epoch,
           prefix + 'iteration': epoch * dataset.num_steps_per_epoch,
           prefix + 'h/':h_norm_dict,
           prefix + 'dh/': dh_norm_dict,}
    if epoch % args.log_dh_interval == 0:
        log[prefix + 'tmp_dh/'] = int_dh_norm_dict
        tmp_pre_act_dict = fetch_h(model, args.multihead)
    if args.wandb:
        wandb.log(log)

def linear_weight_delta( model, linear_model):
    norm_layer_dic, abs_norm_layer_dic, spectral_norm_layer_dic = get_weight_norm_delta(model, linear_model)
    mtensor = get_model_parameters_tensor(model)
    lmtensor = get_model_parameters_tensor(linear_model)
    log = {'width':args.width,
           'epoch':args.epochs,
           'linear_dif/l2_linear':torch.norm(lmtensor),
           'linear_dif/l2_model':torch.norm(mtensor),
           'linear_dif/l2_all':torch.norm(mtensor - lmtensor) / torch.norm(mtensor),
           'linear_dif/abs_all':torch.abs(mtensor - lmtensor).mean(dtype=torch.float32).item() / torch.abs(mtensor).mean(dtype=torch.float32).item(),
           'linear_dif_layer/l2_':norm_layer_dic,
           'linear_dif_layer/abs_':abs_norm_layer_dic,
           'linear_dif_layer/spectral_':spectral_norm_layer_dic}
    if args.wandb:
        wandb.log(log)

def delta_ntk(epoch, model, initial_model, dataset, multihead):
    model.zero_grad()
    initial_model.zero_grad()
    for batch_idx, (x, t) in enumerate(dataset.train_loader):
        model.train()
        initial_model.train()
        x, t = x.to(device), t.to(device)
        t -= args.task1_class_head
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

        if multihead:
            y = model(x, task=1)
            y2 = initial_model(x, task=1)
        else:
            y = model(x)
            y2 = initial_model(x)
        loss1 = loss_func(y,t2)
        loss1.backward()
        loss2 = loss_func(y2,t2)
        loss2.backward()
        ntk = empirical_class_wise_direct_ntk(model, x)
        ntk_init = empirical_class_wise_direct_ntk(initial_model, x)
        model.zero_grad()
        initial_model.zero_grad()
        ntk_del = torch.norm(ntk-ntk_init) / torch.norm(ntk_init) 
        wandb.log({
            'epoch':epoch,
            'ntk_del':ntk_del
        })
        return ntk_del

class ParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)

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

    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--pretrained_lr', type=float, default=1e-1,
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
    parser.add_argument('--ckpt_folder', type=str, default='./ckpts/premutate_mlp_mnist/')
    parser.add_argument('--pretrained_ckpt_folder', type=str, default='./ckpts/mlp_ln_split_cifar_width/')
    parser.add_argument('--load_from_pretrained_ckpt_folder', action='store_true', default=False)

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
    parser.add_argument('--pretrained_parametrization', type=str, default='SP')

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
    parser.add_argument('--train_size', type=int, default=-1)
    parser.add_argument('--pretrained_train_size', type=int, default=-1)

    parser.add_argument('--widen_factor', type=int, default=4)
    parser.add_argument('--loss_type', type=str, default='cross_entropy')

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
    parser.add_argument('--linear_training', action='store_true', default=False)

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
        pretrained_dataset = utils.dataset.MNIST(args=args)
        dataset = utils.dataset.MNIST(args=args, permutate=args.permutate)
    elif args.dataset == 'FashionMNIST':
        pretrained_dataset = utils.dataset.FashionMNIST(args=args)
        dataset = utils.dataset.FashionMNIST(args=args)
    elif args.dataset == 'CIFAR10':
        pretrained_dataset = utils.dataset.CIFAR10(args=args, task_classes=args.pretrained_classes)
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
        if args.load_from_pretrained_ckpt_folder:
            file_name = str(args.model) + '_' + str(args.dataset)  + '_wid_' + str(args.width) + '_ep_' + str(args.pretrained_epochs) + '_param_' + str(args.pretrained_parametrization) + '_tsize_' + str(args.pretrained_train_size) + '_lr_' + str(args.pretrained_lr) + '_loss_' + str(args.loss_type) + '_act_' + str(args.activation) + '.pt'
            folder_path = os.path.join(args.pretrained_ckpt_folder, file_name)
        else:
            file_name = str(args.model) + '_' + str(args.dataset)  + '_wid_' + str(args.width) + '_ep_' + str(args.pretrained_epochs) + '_hp_' + str(args.head_init_epochs) +'_pretrained_param_' + str(args.pretrained_parametrization)+'_param_' + str(args.parametrization) + '_tsize_' + str(args.train_size) + '_lr_' + str(args.pretrained_lr) + '_loss_' + str(args.loss_type) + '_act_' + str(args.activation) + '.pt'
            folder_path = os.path.join(args.ckpt_folder, file_name)
        checkpoint = torch.load(folder_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.population_coding:
            pretrained_orthogonal_matrix = checkpoint['pretrained_orthogonal_matrix'][:args.task1_class_head, :]
            orthogonal_matrix = checkpoint['orthogonal_matrix'][:args.task2_class_head, :]

    initial_model = copy.deepcopy(model)
    if args.linear_training:
        model = LinearizedModel(model)
        if args.parametrization == 'Spectral' or args.parametrization == 'Spectral_output_zero':
            optimizer = create_spectral_optimizer(args, model, lr = args.lr)
        else:
            optimizer = create_optimizer(args, model, lr = args.lr)
        scheduler=None
        if args.head_init_epochs == -1:
            if args.head_init_iterations != -1:
                args.head_init_epochs = 1 + args.head_init_iterations // dataset.num_steps_per_epoch
            elif args.head_init_iterations == -1:
                args.head_init_epochs = 0
        if args.scheduler == 'CosineAnnealingLR':
            scheduler=CosineLRScheduler(optimizer, t_initial=args.epochs,lr_min=0, warmup_t=args.warmup_epochs)
        elif args.scheduler == 'ExponentialLR':
            scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: args.lr * (0.95 ** epoch))
        elif args.scheduler == 'Fraction':
            scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: args.lr / (epoch+1))
        elif args.scheduler == 'PolynomialLR':
            scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=args.sched_power)
            if args.warmup_epochs>0:
                scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epochs, after_scheduler=scheduler)

        if args.log_h_delta:
            for i, data in enumerate(dataset.val_loader, 0):
                inputs, labels = data
                inputs_for_dh = inputs.to(device)
                break
            init_pre_act_dict = fetch_h(model, args.multihead)
            tmp_pre_act_dict  = fetch_h(model, args.multihead)
        try:
            main(epochs=args.epochs, iterations=-1, prefix='LinearTraining/', linear_training=True)
        except ValueError as e:
            print(e)

        linear_model = copy.deepcopy(model)
        model = copy.deepcopy(initial_model)



    if args.log_weight_delta:
        initial_params = [param.clone() for param in model.parameters()]
    if args.parametrization == 'Spectral' or args.parametrization == 'Spectral_output_zero':
        optimizer = create_spectral_optimizer(args, model, lr = args.lr)
    else:
        optimizer = create_optimizer(args, model, lr = args.lr)
    scheduler=None
    if args.head_init_epochs == -1:
        if args.head_init_iterations != -1:
            args.head_init_epochs = 1 + args.head_init_iterations // dataset.num_steps_per_epoch
        elif args.head_init_iterations == -1:
            args.head_init_epochs = 0
    if args.scheduler == 'CosineAnnealingLR':
        scheduler=CosineLRScheduler(optimizer, t_initial=args.epochs,lr_min=0, warmup_t=args.warmup_epochs)
    elif args.scheduler == 'ExponentialLR':
        scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: args.lr * (0.95 ** epoch))
    elif args.scheduler == 'Fraction':
        scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: args.lr / (epoch+1))
    elif args.scheduler == 'PolynomialLR':
        scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=args.sched_power)
        if args.warmup_epochs>0:
            scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epochs, after_scheduler=scheduler)

    if args.log_h_delta:
        for i, data in enumerate(dataset.val_loader, 0):
            inputs, labels = data
            inputs_for_dh = inputs.to(device)
            break
        init_pre_act_dict = fetch_h(model, args.multihead)
        tmp_pre_act_dict  = fetch_h(model, args.multihead)
    try:
        main(epochs=args.epochs, iterations=-1, prefix='')
        if args.linear_training:
            linear_weight_delta( model, linear_model)
    except ValueError as e:
        print(e)
    
    
    wandb.finish()