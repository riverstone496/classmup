'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

def scale_cfg(cfg, scale_factor=1):
    new_cfg = []
    for x in cfg:
        if x == 'M':
            new_cfg.append('M')
        else:
            new_cfg.append(int(x * scale_factor / 4))
    return new_cfg

base_cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG19', scale_factor=1, num_classes=10, num_channels=3, base_width=16):
        super(VGG, self).__init__()
        self.base_width = (base_width/16)
        scaled_cfg = scale_cfg(base_cfg[vgg_name], scale_factor)
        self.scale_factor=scale_factor
        self.input_layer = nn.Conv2d(num_channels, 16*scale_factor, kernel_size=3, padding=1)
        self.features = self._make_layers(scaled_cfg)
        self.output_layer = nn.Linear(scaled_cfg[-2], num_classes)  # Assuming the last non-'M' entry is the size of the final layer
        
        self.input_layer.base_fan_in = 9*num_channels
        self.input_layer.base_fan_out = self.base_width*9*16
        self.output_layer.base_fan_in = self.base_width*base_cfg[vgg_name][-2] // 4
        self.output_layer.base_fan_out = num_classes

    def forward(self, x):
        out = self.input_layer(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.output_layer(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 16*self.scale_factor
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                layers[-3].base_fan_in = 9*self.base_width*in_channels/self.scale_factor
                layers[-3].base_fan_out= 9*self.base_width*x/self.scale_factor
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG('VGG11', scale_factor=2)  # 2倍のchannel数でVGG11を作成
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# テスト関数を呼び出す
test()
