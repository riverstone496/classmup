import torch

import torch
from torch.optim import SGD, Adam

def create_optimizer(args, model, lr, input_lr_const=1):
    # レイヤーごとにパラメータを設定
    param_groups = []
    for name, param in model.named_parameters():
        # レイヤー名に応じて学習率を調整
        layer_lr = lr
        if 'input' in name:
            layer_lr *= input_lr_const * (args.base_width / args.width) ** args.c_input
        elif 'output' in name or 'head' in name:
            layer_lr *= (args.base_width / args.width) ** args.c_output
        param_groups.append({'params': param, 'lr': layer_lr})
    # オプティマイザの選択と初期化
    if args.optim == 'sgd':
        optimizer = SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(args.optim))

    return optimizer

def create_optimizer_for_head(args, model, lr, input_lr_const=1):
    param_groups = []
    for name, param in model.named_parameters():
        if 'output' in name or 'head' in name or 'fc'  in name:
            print(name, 'in head optimizer')
            param_groups.append({'params': param, 'lr': lr})

    # オプティマイザの選択と初期化
    if args.optim == 'sgd':
        optimizer = SGD(param_groups, momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = Adam(param_groups)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(args.optim))

    return optimizer