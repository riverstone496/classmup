
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['mlp']

class MLP(nn.Module):
    def __init__(self, n_hid=1000, nonlin=torch.relu, img_size=28, num_classes=10, num_channels=1, bias=False, depth=3):
        super().__init__()
        self.img_size=img_size
        self.num_channels=num_channels
        self.nonlin=nonlin
        self.depth = depth

        self.input_layer = nn.Linear(img_size*img_size*num_channels, n_hid,bias=bias)
        self.hidden_layer = nn.Linear(n_hid, n_hid,bias=bias)
        self.output_layer = nn.Linear(n_hid, num_classes,bias=bias)

        self.input_layer.base_fan_in = img_size*img_size*num_channels
        self.hidden_layer.base_fan_in = 64
        self.output_layer.base_fan_in = 64
        self.input_layer.base_fan_out = 64
        self.hidden_layer.base_fan_out = 64
        self.output_layer.base_fan_out = num_classes
        self.num_features = n_hid

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.nonlin(self.input_layer(x))
        if self.depth!=2:
            x = self.nonlin(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

class MLPWithBatchNorm(nn.Module):
    def __init__(self, img_size=28, hidden_dim=100, num_classes=10):
        super(MLPWithBatchNorm, self).__init__()

        # First layer with BatchNormalization
        self.fc1 = nn.Linear(img_size*img_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()

        # Second layer with BatchNormalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Output layer
        x = self.fc3(x)

        return x

def mlp(**kwargs):
    model = MLP(**kwargs)
    return model