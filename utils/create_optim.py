import torch
from torch.optim import SGD, Adam

def create_optimizer(args, model, lr, input_lr_const=1, output_lr_const=1):
    # レイヤーごとにパラメータを設定
    param_groups = []
    for name, param in model.named_parameters():
        # レイヤー名に応じて学習率を調整
        layer_lr = lr
        if 'input' in name or 'patch_embed' in name:
            layer_lr *= input_lr_const * (args.base_width / args.width) ** args.c_input
        elif 'output' in name or 'head' in name:
            layer_lr *= output_lr_const * (args.base_width / args.width) ** args.c_output
        else:
            layer_lr *= (args.base_width / args.width) ** args.c_hidden
        print(f"{name} 's learning rate = {layer_lr}")
        param_groups.append({'params': param, 'lr': layer_lr})
    # オプティマイザの選択と初期化
    if args.optim == 'sgd':
        optimizer = SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(args.optim))

    return optimizer

def create_optimizer_for_head(args, model, lr):
    param_groups = []
    for name, param in model.named_parameters():
        if 'block' not in name:
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

def calculate_fan_in_fan_out(name, param):
    if param.dim() == 4:  # 畳み込み層の重みの場合 (出力チャネル数, 入力チャネル数, 高さ, 幅)
        fan_in = param.size(1) * param.size(2) * param.size(3)
        fan_out = param.size(0) * param.size(2) * param.size(3)
    elif param.dim() == 2:  # 全結合層の重みの場合 (出力特徴数, 入力特徴数)
        fan_in = param.size(1)
        fan_out = param.size(0)
    else:
        # その他の層や未対応のケース
        print(f"This function only supports conv2d and linear layers. Not supported for {name}")
        return 1, 1
        #raise NotImplementedError("This function only supports conv2d and linear layers")
    return fan_in, fan_out

def create_spectral_optimizer(args, model, lr):
    # レイヤーごとにパラメータを設定
    param_groups = []
    for name, param in model.named_parameters():
        # レイヤー名に応じて学習率を調整
        layer_lr = lr
        if param.requires_grad:
            if 'weight' in name and 'norm' not in name:
                fan_in, fan_out = calculate_fan_in_fan_out(name, param)
                layer_lr *= (fan_in / fan_out)
        param_groups.append({'params': param, 'lr': layer_lr})
    # オプティマイザの選択と初期化
    if args.optim == 'sgd':
        optimizer = SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(args.optim))

    return optimizer