from .mlp import MLP, MLPWithBatchNorm, MLP_LN
from .cnn import SimpleCNN
from .myrtle import MyrtleNet
from .resnet import ResNet8, ResNet18, ResNet50
from .resnet_nopool import ResNet18NoPool
from .vgg import VGG
import torch
import torch.nn as nn
import numpy as np
from .wideresnet import WideResNet
import timm
from .fractal import create_fractal_model

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
            nn.init.normal_(m.weight.data, mean=0, std=embed_std / (fan_in**0.5))
        if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
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
    elif args.model == 'mlp_ln':
        if args.activation == 'tanh':
            model = MLP_LN(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=torch.tanh, depth=args.depth)
        if args.activation == 'relu':
            model = MLP_LN(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=torch.relu, depth=args.depth)
        if args.activation == 'gelu':
            model = MLP_LN(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=torch.nn.functional.gelu, depth=args.depth)
        if args.activation == 'identity':
            model = MLP_LN(n_hid=args.width,img_size=img_size,num_channels=num_channels,num_classes=num_classes,bias=args.bias, nonlin=lambda x: x, depth=args.depth)
    elif args.model=='MLPWithBatchNorm':
        model = MLPWithBatchNorm(img_size=img_size, hidden_dim=args.width, num_classes=num_classes)
    elif args.model == 'cnn':
        model = SimpleCNN(width=args.width, num_channels=num_channels, input_size=img_size, base_width=args.base_width, num_classes=num_classes, activation=args.activation)
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

def create_finetune_model(num_classes, args):
    model = timm.create_model(model_name = args.model, pretrained=True, num_classes= num_classes)
    if hasattr(model, 'head'):
        m = model.head
    elif hasattr(model, 'fc'):
        m = model.fc
    if args.output_nonzero:
        nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')
    else:
        nn.init.zeros_(m.weight.data)
    return model

class MultiHeadModel(nn.Module):
    def __init__(self, args, img_size, num_classes, num_channels):
        super(MultiHeadModel, self).__init__()
        self.default_task = 0
        if args.use_cifar_model:
            self.base_model = create_model(img_size, num_classes, num_channels, args)
        elif args.use_fractal_model:
            self.base_model = create_fractal_model(model_name = args.model, pretrained=True, pretrained_path=args.pretrained_path)
        else:
            self.base_model = timm.create_model(model_name = args.model, pretrained=True)
        if hasattr(self.base_model, 'head'):
            self.base_model.head = nn.Identity()  # 元のheadを除去
        elif hasattr(self.base_model, 'fc'):
            self.base_model.fc = nn.Identity()  # 元のheadを除去
        elif hasattr(self.base_model, 'output_layer'):
            self.base_model.output_layer = nn.Identity()  # 元のheadを除去
        self.head1 = nn.Linear(self.base_model.num_features, args.task1_class)  # Task 1のhead
        self.head2 = nn.Linear(self.base_model.num_features, args.task2_class)  # Task 2のhead

        if 'zero' in args.task1_parametrization:
            nn.init.zeros_(self.head1.weight.data)
        else:
            nn.init.kaiming_normal_(self.head1.weight.data, a=1, mode='fan_in')
            if 'muP' in args.task1_parametrization or 'Spectral' in args.task1_parametrization:
                self.head1.weight.data /= (self.base_model.num_features / (args.task1_class))**(1/2)
        if 'zero' in args.task2_parametrization:
            nn.init.zeros_(self.head2.weight.data)
        else:
            nn.init.kaiming_normal_(self.head2.weight.data, a=1, mode='fan_in')
            if 'muP' in args.task2_parametrization or 'Spectral' in args.task2_parametrization:
                self.head2.weight.data /= (self.base_model.num_features / (args.task2_class))**(1/2)            

    def forward(self, x, task=None):
        if task == None:
            task = self.default_task
        x = self.base_model(x)
        if task == 0:
            x = self.head1(x)
        elif task == 1:
            x = self.head2(x)
        return x
    
    def disable_head1_grad(self):
        """
        head1の全てのパラメータに対して、勾配計算を無効にする。
        """
        for param in self.head1.parameters():
            param.requires_grad = False
