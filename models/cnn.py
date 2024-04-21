import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, width=16, num_channels=3, input_size=28, base_width=8, num_classes=10):
        super(SimpleCNN, self).__init__()
        filters1, filters2 = width, 2 * width
        self.input_layer = nn.Conv2d(num_channels, filters1, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(filters1, filters2, 5, stride=1, padding=2)
        self.output_layer = nn.Linear(filters2 * (input_size//4) * (input_size//4), num_classes)
        
        ### base (when width=8)
        self.input_layer.base_fan_in=25*num_channels
        self.conv2.base_fan_in=25*base_width
        self.output_layer.base_fan_in=2 * base_width * (input_size//4) * (input_size//4)
        self.input_layer.base_fan_out=25*base_width
        self.conv2.base_fan_out=50*base_width
        self.output_layer.base_fan_out=10
        ###
        self.num_features = filters2 * (input_size//4) * (input_size//4)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # flatten
        x = self.output_layer(x)
        return x