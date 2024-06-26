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
import utils.continual_dataset
from models.create_model import create_model,initialize_weight, create_finetune_model, MultiHeadModel
from utils.damping import set_damping
import wandb
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR, PolynomialLR
from utils.loss_type import CustomCrossEntropyLoss, CustomMSELossWithL2Reg
from utils.create_optim import create_optimizer, create_optimizer_for_head, create_spectral_optimizer
import warmup_scheduler

dataset_options = ['MNIST','CIFAR10','CIFAR100','SVHN','Flowers','Cars', 'FashionMNIST', 'STL10']

max_validation_acc={0:0, 1:0}
min_validation_loss={0:np.inf, 1:np.inf}
max_train_acc={0:0, 1:0}
min_train_loss={0:np.inf, 1:np.inf}
max_train_acc_all={0:0, 1:0}
min_train_loss_all={0:np.inf, 1:np.inf}
prev_epochs = {'init_':0, '':0}
job_id = os.environ.get('SLURM_JOBID')
if job_id is not None:
    os.environ["WANDB_HOST"] = job_id

def main(epochs, iterations = -1, prefix = '', task_index = 0):
    global task0_epoch
    total_train_time=0
    epoch = 0
    for i in range(task_index+1):
        train_accuracy=trainloss_all(epoch, prefix, task_index=i, phase=task_index)
        nantf = val(epoch, prefix, task_index=i, phase=task_index)
    for epoch in range(1, epochs + 1):
        #epoch += task_index * prev_epochs[prefix]
        start = time.time()
        train(epoch, prefix, iterations, task_index=task_index)
        total_train_time += time.time() - start
        if (epoch-1)%args.log_val_interval==0:
            for i in range(task_index+1):
                train_accuracy=trainloss_all(epoch, prefix, task_index=i, phase=task_index)
                nantf = val(epoch, prefix, task_index=i, phase=task_index)
            if args.log_h_delta:
                log_h_delta(epoch, prefix, phase=task_index)
            if nantf:
                break
            if args.train_acc_stop is not None and train_accuracy > args.train_acc_stop:
                wandb.run.summary['total_epochs_task'+str(task_index)] = epoch
                break
    prev_epochs[prefix] += epoch
    print(f'total_train_time: {total_train_time:.2f}s')
    if args.epochs!= 0 :
        print(f'avg_epoch_time: {total_train_time / args.epochs:.2f}s')
        print(f'avg_step_time: {total_train_time / args.epochs / dataset.num_steps_per_epoch * 1000:.2f}ms')
    if args.wandb:
        wandb.run.summary['total_train_time'] = total_train_time
        if args.epochs != 0:
            wandb.run.summary['avg_epoch_time'] = total_train_time / args.epochs
            wandb.run.summary['avg_step_time'] = total_train_time / args.epochs / dataset.num_steps_per_epoch

def val(epoch, prefix = '', task_index = 0, phase = 0):
    global max_validation_acc,min_validation_loss
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset.val_loader[task_index]:
            data, target = data.to(device), target.to(device)
            if task_index == 1:
                target -= args.task1_class
            output = model(data, task_index)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataset.val_loader[task_index].dataset)
    test_accuracy = 100. * correct / len(dataset.val_loader[task_index].dataset)
    if test_accuracy>max_validation_acc[task_index]:
        max_validation_acc[task_index]=test_accuracy
    if test_loss<min_validation_loss[task_index]:
        min_validation_loss[task_index]=test_loss

    if args.wandb:
        log = {prefix+'epoch': epoch,
               prefix+'iteration': epoch * dataset.num_steps_per_epoch,
               prefix+'task'+ str(task_index)+ '_val_loss': test_loss,
               prefix+'task'+ str(task_index)+ '_val_accuracy': test_accuracy,
               prefix+'task'+ str(task_index)+ '_max_validation_acc':max_validation_acc[task_index],
               prefix+'task'+ str(task_index)+ '_min_validation_loss':min_validation_loss[task_index]}
        wandb.log(log)
        log = {prefix+'epoch': epoch,
               'task'+str(phase)+'/'+prefix+'task'+ str(task_index)+ '_val_loss': test_loss,
               'task'+str(phase)+'/'+prefix+'task'+ str(task_index)+ '_val_accuracy': test_accuracy}
        wandb.log(log)
    if task_index == 0:
        wandb.run.summary['task0_val_accuracy_final'] = max_validation_acc[task_index]
    print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(dataset.val_loader[task_index].dataset), test_accuracy))

    if math.isnan(test_loss):
        print('Error: Train loss is nan', file=sys.stderr)
        return True
    return False

def trainloss_all(epoch, prefix = '', task_index = 0, phase=0):
    global max_train_acc_all,min_train_loss_all
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset.train_val_loader[task_index]:
            data, target = data.to(device), target.to(device)
            if task_index == 1:
                target -= args.task1_class
            output = model(data, task_index)
            train_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(dataset.train_val_loader[task_index].dataset)
    train_accuracy = 100. * correct / len(dataset.train_val_loader[task_index].dataset)
    
    if train_accuracy>max_train_acc_all[task_index]:
        max_train_acc_all[task_index]=train_accuracy
    if train_loss<min_train_loss_all[task_index]:
        min_train_loss_all[task_index]=train_loss

    if args.wandb:
        log = {prefix+'epoch': epoch,
               prefix+'iteration': (epoch) * dataset.num_steps_per_epoch,
               prefix+'task' + str(task_index) +'_train_loss_all': train_loss,
               prefix+'task' + str(task_index) +'_train_accuracy_all': train_accuracy,
               prefix+'task' + str(task_index) +'_max_train_acc_all':max_train_acc_all[task_index],
               prefix+'task' + str(task_index) +'_min_train_loss_all':min_train_loss_all[task_index]}
        wandb.log(log)
        log = {prefix+'epoch': epoch,
               'task'+str(phase)+'/'+prefix+'task' + str(task_index) +'_train_loss_all': train_loss,
               'task'+str(phase)+'/'+prefix+'task' + str(task_index) +'_train_accuracy_all': train_accuracy}
        wandb.log(log)
    print('Train all set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(dataset.val_loader[task_index].dataset), train_accuracy))

    if math.isnan(train_loss):
        print('Error: Train loss is nan', file=sys.stderr)
        return train_accuracy
    return train_accuracy

def train(epoch, prefix = '', train_iterations=-1, task_index = 0, phase=0):
    global max_train_acc,min_train_loss
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (x, t) in enumerate(dataset.train_loader[task_index]):
        if train_iterations != -1 and (epoch-1) * dataset.num_steps_per_epoch+batch_idx >= train_iterations:
            return
        model.train()
        x, t = x.to(device), t.to(device)

        if args.loss_type=='cross_entropy':
            loss_func = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            if task_index == 1:
                t -= args.task1_class
            t2= t
        elif args.loss_type=='mse':
            loss_func = torch.nn.MSELoss()
            t2 = MSE_label(x, t)

        y = model(x, task_index)
        loss = loss_func(y,t2)
        loss.backward()
        grad_norm = get_grad_norm(model)
        if batch_idx%args.accumulate_iters == args.accumulate_iters-1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        if batch_idx%100==0 and args.wandb:
            pred = y.data.max(1)[1]
            acc = 100. * pred.eq(t.data).cpu().sum() / t.size(0)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(dataset.train_loader[task_index].dataset),
                100. * batch_idx / dataset.num_steps_per_epoch, float(loss)))
            l1_norm, l2_norm = get_weight_norm(model)
            layer_weight_norm(epoch, model, prefix = prefix)
            if acc>max_train_acc[task_index]:
                max_train_acc[task_index]=acc
            if loss<min_train_loss[task_index]:
                min_train_loss[task_index]=loss
            log = {prefix+'epoch': epoch,
                   prefix+'iteration': (epoch-1) * dataset.num_steps_per_epoch+batch_idx,
                   prefix+'task'+str(task_index) +'_train_loss': float(loss),
                   prefix+'task'+str(task_index) +'_train_accuracy': float(acc),
                   prefix+'task'+str(task_index) +'_max_train_acc':max_train_acc[task_index],
                   prefix+'task'+str(task_index) +'_min_train_loss':min_train_loss[task_index],
                   prefix+'task'+str(task_index) +'_l1_norm':l1_norm,
                   prefix+'task'+str(task_index) +'_l2_norm':l2_norm,
                   prefix+'task'+str(task_index) +'_grad_norm_all':grad_norm
                   }
            wandb.log(log)

    if scheduler is not None:
        scheduler.step(epoch=epoch)

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
    y = model(inputs_for_dh, 0)
    y = model(inputs_for_dh, 1)
    pre_act_dict = {}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
              continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        if not hasattr(module, 'out_data'):
            continue
        pre_act_dict[name] = module.out_data
    return pre_act_dict

def layer_weight_norm(epoch, model, prefix = ''):
    weight_dict = {}
    grad_dict = {}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
              continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        if module.weight.grad is None:
            continue
        weight_dict[name] = torch.abs(module.weight.data).mean(dtype=torch.float32).item()
        grad_dict[name] = torch.abs(module.weight.grad.data).mean(dtype=torch.float32).item()
    log = {'epoch': epoch,
           prefix + 'weight_norm/':weight_dict,
           prefix + 'grad_norm/': grad_dict,}
    if args.wandb:
        wandb.log(log)

def log_h_delta(epoch, prefix = '', phase = ''):
    pre_act_dict = fetch_h(model)
    h_norm_dict = {}
    dh_norm_dict = {}
    for mname in pre_act_dict.keys():
        h_norm_dict[mname] = torch.abs(pre_act_dict[mname]).mean(dtype=torch.float32).item()
        dh_norm_dict[mname] = torch.abs(pre_act_dict[mname] - init_pre_act_dict[mname]).mean(dtype=torch.float32).item()
    if args.wandb:
        log = {prefix + 'epoch': epoch,
               prefix + 'iteration': epoch * dataset.num_steps_per_epoch,
               'task'+str(phase)+ '_' + prefix + 'h/':h_norm_dict,
               'task'+str(phase)+ '_' + prefix + 'dh/': dh_norm_dict,}
        wandb.log(log)

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
        args.output_nonzero = True
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
    if args.parametrization == 'UP_output_zero':
        args.output_nonzero = False
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
        args.output_nonzero = True
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
    if args.parametrization == 'Spectral':
        args.output_nonzero = True
        args.b_output=1/2
        args.b_hidden=1/2
        args.b_input=1/2
    if args.parametrization == 'Spectral_output_zero':
        args.output_nonzero = False
        args.b_output=1/2
        args.b_hidden=1/2
        args.b_input=1/2
    if args.parametrization == 'Spectral_LR':
        args.output_nonzero = True
        args.b_output=1
        args.b_hidden=1/2
        args.b_input=1/2
    if args.parametrization == 'Spectral_LR_output_zero':
        args.output_nonzero = False
        args.b_output=1
        args.b_hidden=1/2
        args.b_input=1/2

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='CIFAR10',
                        choices=dataset_options)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--base_width', type=int, default=1)
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
    parser.add_argument('--epochs_2', type=int, default=None,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--train_acc_stop', type=float, default=None,
                        help='train_acc_stop (default: 20)')
    parser.add_argument('--lp_epochs', type=int, default=-1,
                        help='number of iterations to train head (default: 0)')
    parser.add_argument('--lp_epochs_2', type=int, default=None,
                        help='number of iterations to train head (default: 0)')
    parser.add_argument('--lp_iterations', type=int, default=-1,
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

    parser.add_argument('--learning_rate1', type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--learning_rate2', type=float, default=1e-1,
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
    
    parser.add_argument('--task1_parametrization', type=str, default=None)
    parser.add_argument('--task2_parametrization', type=str, default='SP')
    parser.add_argument('--output_nonzero', action='store_true', default=False)
    parser.add_argument('--curvature_update_interval', type=int, default=1)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--scheduler_2', type=str, default='Constant')
    parser.add_argument('--sched_power', type=float, default=1,
                        help='sched_power')
    parser.add_argument('--warmup_epochs1', type=int, default=0,
                        help='sched_power')
    parser.add_argument('--warmup_epochs2', type=int, default=0,
                        help='sched_power')
    
    parser.add_argument('--resume_ckpo_folder', type=str, default=None,
                        help='sched_power')
    

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
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--train_size', type=int, default=-1)
    parser.add_argument('--train_size_2', type=int, default=None)

    parser.add_argument('--widen_factor', type=int, default=4)
    parser.add_argument('--task1_class', type=int, default=5)
    parser.add_argument('--loss_type', type=str, default='cross_entropy')

    parser.add_argument('--log_record', action='store_true', default=False)
    parser.add_argument('--use_timm', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--wandb', action='store_false', default=True)

    parser.add_argument('--withoutShortcut', action='store_true', default=False)
    parser.add_argument('--withoutBN', action='store_true', default=False)
    parser.add_argument('--class_scaling', action='store_true', default=False)
    parser.add_argument('--use_cifar_model', action='store_true', default=False)

    parser.add_argument('--use_fractal_model', action='store_true', default=False)
    parser.add_argument('--pretrained_path', type=str, default='./pretrained/exfractal_21k_tiny.pth.tar')

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

    args.lr1 = args.learning_rate1
    args.lr2 = args.learning_rate2
    args.head_init_iterations = args.lp_iterations

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
    
    print('Start creating Dataset')
    if args.dataset == 'MNIST':
        dataset = utils.continual_dataset.MNIST(args=args)
    elif args.dataset == 'FashionMNIST':
        dataset = utils.continual_dataset.FashionMNIST(args=args)
    elif args.dataset == 'CIFAR10':
        dataset = utils.continual_dataset.CIFAR10(args=args)
    elif args.dataset == 'CIFAR100':
        dataset = utils.continual_dataset.CIFAR100(args=args)
    elif args.dataset == 'STL10':
        dataset = utils.continual_dataset.STL(args=args)
    elif args.dataset == 'SVHN':
        dataset = utils.continual_dataset.SVHN(args=args)
    elif args.dataset == 'Flowers':
        dataset = utils.continual_dataset.Flowers(args=args)
    elif args.dataset == 'Cars':
        dataset = utils.continual_dataset.Cars(args=args)
    print('Finished creating Dataset')

    if args.class_scaling:
        if args.model == 'mlp':
            dataset.num_classes = args.width

    if args.pseudo_batch_size != -1:
        args.accumulate_iters = args.pseudo_batch_size / args.batch_size
    else:
        args.accumulate_iters=1
        
    if args.lp_epochs == -1:
        if args.lp_iterations != -1:
            args.lp_epochs = 1 + args.lp_iterations // dataset.num_steps_per_epoch
        elif args.lp_iterations == -1:
            args.lp_epochs = 0

    if args.pseudo_batch_size != -1:
        args.batch_size=args.pseudo_batch_size
    print('Start Model Preparing')
    model = MultiHeadModel(args=args, num_classes = dataset.num_classes).to(device=device)
    if args.task2_parametrization is None:
        args.task2_parametrization = args.task1_parametrization

    for task_num in range(2):
        if task_num ==0 and args.epochs == 0:
            continue
        if task_num == 1:
            model.disable_head1_grad()
        if task_num == 0 and args.resume_ckpo_folder is not None:
            file_name = str(args.model) + '_wid_' + str(args.width) + '_ep_' + str(args.epochs) +'_hp_' + str(args.lp_epochs) + '_' + str(args.task1_parametrization) + '_ep' + str(args.epochs) + '_lr1_' + str(args.learning_rate1) + '.pt'
            folder_path = os.path.join(args.resume_ckpo_folder, file_name)
            if os.path.isfile(folder_path):
                checkpoint = torch.load(folder_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                continue
            
        print('start task=',task_num)
        if task_num==0:
            learning_rate = args.learning_rate1
            warmup_epochs = args.warmup_epochs1
            args.parametrization = args.task1_parametrization
        elif task_num==1:
            learning_rate = args.learning_rate2
            warmup_epochs = args.warmup_epochs2
            args.scheduler = args.scheduler_2
            args.parametrization = args.task2_parametrization
            if args.lp_epochs_2 != None:
                args.lp_epochs = args.lp_epochs_2
            if args.epochs_2 != None:
                args.epochs = args.epochs_2
        muP_set(args)
        # Head_Init_Iters
        if args.log_weight_delta:
            initial_params = [param.clone() for param in model.parameters()]
        optimizer = create_optimizer_for_head(args, model, lr = args.init_lr)
        scheduler=None    
        if args.log_h_delta:
            for i, data in enumerate(dataset.val_loader[0], 0):
                inputs, labels = data
                inputs_for_dh = inputs.to(device)
                break
            init_pre_act_dict = fetch_h(model)
        try:
            main(epochs=args.lp_epochs, iterations=args.lp_iterations, prefix='init_', task_index=task_num)
        except ValueError as e:
            print(e)

        if args.log_weight_delta:
            initial_params = [param.clone() for param in model.parameters()]
        if 'Spectral' in args.parametrization:
            optimizer = create_spectral_optimizer(args, model, lr = learning_rate)
        else:
            optimizer = create_optimizer(args, model, lr = learning_rate)
        scheduler=None
        if args.lp_epochs == -1:
            if args.lp_iterations != -1:
                args.lp_epochs = 1 + args.lp_iterations // dataset.num_steps_per_epoch
            elif args.lp_iterations == -1:
                args.lp_epochs = 0
        if args.scheduler == 'CosineAnnealingLR':
            scheduler=CosineLRScheduler(optimizer, t_initial=args.epochs,lr_min=0, warmup_t=warmup_epochs)
        elif args.scheduler == 'ExponentialLR':
            scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: args.lr * (0.95 ** epoch))
        elif args.scheduler == 'Fraction':
            scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: args.lr / (epoch+1))
        elif args.scheduler == 'PolynomialLR':
            scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=args.sched_power)
            if args.warmup_epochs>0:
                scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epochs, after_scheduler=scheduler)

        if args.log_h_delta:
            for i, data in enumerate(dataset.val_loader[0], 0):
                inputs, labels = data
                inputs_for_dh = inputs.to(device)
                break
            init_pre_act_dict = fetch_h(model)
        try:
            main(epochs=args.epochs, iterations=-1, prefix='', task_index=task_num)
        except ValueError as e:
            print(e)

        if task_num == 0 and args.resume_ckpo_folder is not None:
            file_name = str(args.model) + '_wid_' + str(args.width) + '_ep_' + str(args.epochs) +'_hp_' + str(args.lp_epochs) + '_' + str(args.task1_parametrization) + '_ep' + str(args.epochs) + '_lr1_' + str(args.learning_rate1) + '.pt'
            folder_path = os.path.join(args.resume_ckpo_folder, file_name)
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, folder_path)
            
    
    wandb.finish()