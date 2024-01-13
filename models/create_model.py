from .mlp import MLP, MLPWithBatchNorm
from .cnn import SimpleCNN
from .myrtle import MyrtleNet
from .resnet import ResNet8, ResNet18, ResNet50
from .resnet_nopool import ResNet18NoPool
from .vgg import VGG
import torch
import torch.nn as nn
import numpy as np
from .wideresnet import WideResNet

def calculate_fan_in_fan_out(module):
    parameters = list(module.parameters())
    fan_in = 0
    fan_out = 0

    if isinstance(module, nn.Conv2d):
        # For Convolutional layers
        n_input_feature_maps = parameters[0].shape[1]
        n_output_feature_maps = parameters[0].shape[0]
        receptive_field_size = parameters[0].shape[2] * parameters[0].shape[3]
        fan_in = n_input_feature_maps * receptive_field_size
        fan_out = n_output_feature_maps * receptive_field_size
    elif isinstance(module, nn.Linear):
        # For fully connected layers
        fan_in = parameters[0].shape[1]
        fan_out = parameters[0].shape[0]
    elif isinstance(module, nn.Embedding):
        # For Embedding layers
        fan_in = parameters[0].shape[0]  # Vocabulary size
        fan_out = parameters[0].shape[1]  # Embedding dimension
    
    return fan_in, fan_out

def initialize_weight(model,b_input=0.5,b_hidden=0.5,b_output=0.5,output_nonzero=False,output_var_mult=1,width=None,embed_std=0.01):
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if all(not p.requires_grad for p in m.parameters()):
            continue
        if isinstance(m,nn.BatchNorm2d):
            continue
        
        fan_in, fan_out = calculate_fan_in_fan_out(m)
        if hasattr(m,'base_fan_in'):
            fan_in = fan_in/m.base_fan_in
            fan_out = fan_out/m.base_fan_out
        else:
            SystemError('Error base fan in is not set')

        if width is not None:
            if 'input' in name or 'block1.layer.0.conv1.weight' in name:
                fan_in = 1
                fan_out = width
            elif 'output' in name:
                fan_in = width
                fan_out = 1
            else:
                fan_in = width
                fan_out = width
        if isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight.data, mean=0, std=embed_std)
        else:
            nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')
        if 'input' in name or 'block1.layer.0.conv1.weight' in name:
            m.weight.data /= fan_in**(b_input-0.5)
        if 'output' in name:
            if not output_nonzero:
                print('init zero')
                nn.init.zeros_(m.weight)
            m.weight.data /= fan_in**(b_output-0.5)
            m.weight.data *= output_var_mult
        else:
            m.weight.data /= fan_in**(b_hidden-0.5)
        if hasattr(m,'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
            m.bias.data /= fan_in**(b_input-0.5)

    return model

def create_model(img_size, num_classes, num_channels, args):
    if args.model == 'mlp':
        if args.activation == 'tanh':
            model = MLP(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=torch.tanh, depth=args.depth)
        if args.activation == 'relu':
            model = MLP(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=torch.relu, depth=args.depth)
        if args.activation == 'gelu':
            model = MLP(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=torch.nn.functional.gelu, depth=args.depth)
        if args.activation == 'identity':
            model = MLP(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=lambda x: x, depth=args.depth)
    elif args.model=='MLPWithBatchNorm':
        model = MLPWithBatchNorm(img_size=img_size, hidden_dim=args.width, num_classes=num_classes)
    elif args.model == 'cnn':
        model = SimpleCNN(width=args.width, num_channels=num_channels, input_size=img_size, base_width=args.base_width, num_classes=num_classes)
    elif args.model == 'myrtle':
        model = MyrtleNet(width=args.width,img_size=img_size, depth=args.depth, num_input_channels=num_channels)
    elif args.model == 'resnet8':
        model = ResNet8(num_classes=num_classes,width=args.width, num_channels=num_channels, withoutShortcut = args.withoutShortcut, withoutBN=args.withoutBN)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes,width=args.width, num_channels=num_channels, withoutShortcut = args.withoutShortcut, withoutBN=args.withoutBN, base_width=args.base_width)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes=num_classes,width=args.width, num_channels=num_channels, withoutShortcut = args.withoutShortcut, withoutBN=args.withoutBN, base_width=args.base_width)
    elif args.model == 'wideresnet10':
        model = WideResNet(depth=10, num_classes=num_classes, widen_factor=args.width,
                             dropRate=0)
    elif args.model == 'wideresnet16':
        model = WideResNet(depth=16, num_classes=num_classes, widen_factor=args.width,
                             dropRate=0)
    elif args.model == 'wideresnet28':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=args.width,
                             dropRate=0)
    elif args.model == 'resnet18_nopool':
        model = ResNet18NoPool(num_classes=num_classes,width=args.width, num_channels=num_channels, withoutShortcut = args.withoutShortcut, withoutBN=args.withoutBN)
    elif args.model == 'vgg19':
        model = VGG(vgg_name='VGG19', scale_factor=args.width, num_classes=num_classes, num_channels=num_channels, base_width=args.base_width)
    return model