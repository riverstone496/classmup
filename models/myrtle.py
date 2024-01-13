import torch
import torch.nn as nn
import math

class MyrtleNet(nn.Module):
    def __init__(self, num_input_channels=3, img_size=32, depth=5, width=1):
        super(MyrtleNet, self).__init__()

        layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}
        factor = layer_factor[depth]

        self.input_layer = nn.Conv2d(in_channels=num_input_channels, 
                                      out_channels=width, 
                                      kernel_size=3, 
                                      stride=1, 
                                      padding=1)
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Conv2d(width, 
                                      out_channels=width, 
                                      kernel_size=3, 
                                      stride=1, 
                                      padding=1), 
                            nn.ReLU()] * (factor[0]-1))
        self.hidden_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.hidden_layers.extend([nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1), 
                            nn.ReLU()] * factor[1])
        self.hidden_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.hidden_layers.extend([nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1), 
                            nn.ReLU()] * factor[2])
        self.hidden_layers.extend([nn.AvgPool2d(kernel_size=2, stride=2)] * int(math.log2(img_size/4)))

        # Compute the size of the flattened features after the Conv and Pooling layers
        conv_output_size = self._get_conv_output_size((num_input_channels, img_size, img_size)) 
        self.hidden_layers.append(nn.Flatten())
        self.output_layer = nn.Linear(conv_output_size, 10)

        self.input_layer.base_fan_in = num_input_channels * 9
        self.input_layer.base_fan_out = 16 * 9
        self.output_layer.base_fan_in = 16
        self.output_layer.base_fan_out = 10
        for layer in self.hidden_layers:
            layer.base_fan_in  = 16*9
            layer.base_fan_out = 16*9

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = self.input_layer(x)
        x = torch.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return x.view(1, -1).size(1)
