import torch
import torch.nn as nn
import copy

class LinearModel(nn.Module):
    def __init__(self, model):
        super(LinearModel, self).__init__()
        self.model = model
        self.initial_model = copy.deepcopy(self.model)
        self.model_parameter = nn.ParameterList([p for p in self.model.parameters()])
        self.initial_model_parameter = nn.ParameterList([p for p in self.initial_model.parameters()])

    def forward(self, x):
        f0 = self.initial_model(x)
        J = self.compute_jacobian(x)
        delta_theta = [param - init_param for param, init_param in zip(self.model_parameter, self.initial_model_parameter)]
        delta_theta = torch.cat([dt.view(-1) for dt in delta_theta])
        J = J.view(-1, delta_theta.size(0))
        f = f0 + torch.matmul(delta_theta, J.T)
        return f

    def compute_jacobian(self, x):
        x = x.requires_grad_(True)
        f0 = self.initial_model(x)
        jacobian = []
        for f0_i in f0:
            grads = torch.autograd.grad(f0_i, x, retain_graph=True, create_graph=True)[0]
            jacobian.append(grads.view(-1))
        jacobian = torch.stack(jacobian)
        return jacobian