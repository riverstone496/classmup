import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.func import vmap, jacrev
from functorch import jvp, make_functional_with_buffers

def empirical_ntk_jacobian_contraction(model, x1, x2, block_size=32):
    func0, params0, buffers0 = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )
    params = [p for p in model.parameters()]
    fnet_single = lambda params, x: func0(params, buffers0, x)
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1_flat = torch.cat([j.reshape(j.shape[0], -1) for j in jac1], dim=1)
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2_flat = torch.cat([j.reshape(j.shape[0], -1) for j in jac2], dim=1)
    # Initialize result matrix
    num_x1 = jac1_flat.shape[0]
    num_x2 = jac2_flat.shape[0]
    result = torch.zeros((num_x1, num_x2), device=jac1_flat.device)
    # Block-wise computation to save memory
    for i in range(0, num_x1, block_size):
        for j in range(0, num_x2, block_size):
            block_jac1 = jac1_flat[i:i+block_size]
            block_jac2 = jac2_flat[j:j+block_size]
            result[i:i+block_size, j:j+block_size] = block_jac1 @ block_jac2.T
    return result
