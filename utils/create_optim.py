import torch

import torch
from torch.optim import SGD, Adam

def create_optimizer(args, model, lr, head_only = False):
    # レイヤーごとにパラメータを設定
    param_groups = []
    for name, param in model.named_parameters():
        # レイヤー名に応じて学習率を調整
        if 'input' in name:
            lr *= (1 / args.width) ** args.c_input
        elif 'output' in name:
            lr *= (1 / args.width) ** args.c_output

        if args.head_only and 'output' not in name:
            lr = 0

        param_groups.append({'params': param, 'lr': lr})

    # オプティマイザの選択と初期化
    if args.optim == 'sgd':
        optimizer = SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(args.optim))

    return optimizer
