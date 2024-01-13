'''ResNet8 for FashionMNIST in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, base_in_width=16, base_width=16, withoutShortcut=False, withoutBN=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            self.shortcut[0].base_fan_in  =  base_in_width
            self.shortcut[0].base_fan_out = self.expansion*base_width
        self.conv1.base_fan_in = 1*base_in_width
        self.conv1.base_fan_out = 1*base_width
        self.conv2.base_fan_in = 9*base_width
        self.conv2.base_fan_out = 9*base_width
        self.conv3.base_fan_in = 1*base_width
        self.conv3.base_fan_out = 4*base_width

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, base_in_width=16, base_width=16, withoutShortcut=False, withoutBN=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.withoutShortcut = withoutShortcut
        self.withoutBN = withoutBN
        if not self.withoutBN:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = None
            self.bn2 = None
        if not self.withoutShortcut:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                modules = [nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)]
                if not self.withoutBN:
                    modules.append(nn.BatchNorm2d(self.expansion*planes))
                self.shortcut = nn.Sequential(*modules)

                self.shortcut[0].base_fan_in  =  base_in_width
                self.shortcut[0].base_fan_out = self.expansion*base_width
        self.conv1.base_fan_in = 9*base_in_width
        self.conv1.base_fan_out = 9*base_width
        self.conv2.base_fan_in = 9*base_width
        self.conv2.base_fan_out = 9*base_width

    def forward(self, x):
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        if not self.withoutShortcut:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=1, num_classes=10, width=1, base_width=16, withoutShortcut=False, withoutBN=False):
        super(ResNet, self).__init__()
        self.in_planes = 16 * width
        self.base_planes = base_width
        self.withoutShortcut = withoutShortcut
        self.withoutBN = withoutBN

        self.input_layer = conv3x3(num_channels, 16 * width)
        self.bn1 = nn.BatchNorm2d(16 * width)
        self.layer1 = self._make_layer(block, 16 * width, num_blocks[0], stride=1 , base_width=base_width)
        self.layer2 = self._make_layer(block, 32 * width, num_blocks[1], stride=2 , base_width=2*base_width)
        self.layer3 = self._make_layer(block, 64 * width, num_blocks[2], stride=2 , base_width=4*base_width)
        self.layer4 = self._make_layer(block, 128 * width, num_blocks[3], stride=2, base_width=8*base_width)
        self.output_layer = nn.Linear(128*block.expansion*width, num_classes)

        self.input_layer.base_fan_in   = 9*num_channels
        self.input_layer.base_fan_out  = 9*base_width
        self.output_layer.base_fan_in  = 8*base_width*block.expansion
        self.output_layer.base_fan_out = num_classes

    def _make_layer(self, block, planes, num_blocks, stride, base_width):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, base_in_width=self.base_planes, base_width=base_width, withoutShortcut=self.withoutShortcut, withoutBN=self.withoutBN))
            self.in_planes = planes * block.expansion
            self.base_planes = base_width * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.input_layer(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])  # Change the size from 4 to the size of the input
        out = out.view(out.size(0), -1)
        out = self.output_layer(out)
        return out


def ResNet8(num_classes=10, num_channels=1, width=1, withoutShortcut = False, withoutBN=False):
    return ResNet(BasicBlock, [1,1,1,1], num_channels, num_classes, width, withoutShortcut=withoutShortcut, withoutBN=withoutBN)

def ResNet18(num_classes=10, num_channels=1, width=1, base_width=16, withoutShortcut = False, withoutBN=False):
    return ResNet(BasicBlock, [2,2,2,2], num_channels, num_classes, width, base_width=base_width, withoutShortcut=withoutShortcut, withoutBN=withoutBN)

def ResNet50(num_classes=10, num_channels=1, width=1, base_width=1, withoutShortcut = False, withoutBN=False):
    return ResNet(Bottleneck, [3,4,6,3], num_channels, num_classes, width, base_width=base_width, withoutShortcut=withoutShortcut, withoutBN=withoutBN)

# Add similar changes to ResNet18, ResNet34 etc.

def test_resnet():
    net = ResNet8()
    y = net(Variable(torch.randn(1,1,28,28)))  # Change input size from (1,3,32,32) to (1,1,28,28)
    print(y.size())
