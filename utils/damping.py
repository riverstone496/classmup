import torch
import torch.nn as nn
import numpy as np

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

def set_damping(model,args):
    #args.dA = -1
    #args.dB = 2 * args.b_output - 1
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
        if 'input' in name or 'block1.layer.0.conv1.weight' in name:
            m.damping_A = args.damping_A
            m.damping_B = args.damping_B/(fan_out**args.dB)
        elif 'output' in name:
            m.damping_A = args.damping_A/(fan_in**args.dA)
            m.damping_B = args.damping_B
        else:
            m.damping_A = args.damping_A/(fan_in**args.dA)
            m.damping_B = args.damping_B/(fan_out**args.dB)
        print(name,fan_in,fan_out,'m.damping_A',m.damping_A,'m.damping_B',m.damping_B)
    return model
