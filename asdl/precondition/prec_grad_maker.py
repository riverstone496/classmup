import math
import warnings
from dataclasses import dataclass
from typing import List, Any, Iterable
from torch import Tensor

import torch
import torch.nn as nn
from .. import GradientMaker
from ..fisher import LOSS_CROSS_ENTROPY,SHAPE_LAYER_WISE,SHAPE_KRON,FISHER_MC
from ..fisher import FisherConfig, get_fisher_maker
from ..vector import ParamVector

__all__ = ['PreconditionedGradientMaker', 'PreconditioningConfig',
           'get_update_schedule', 'INTERVAL_CONSTANT', 'INTERVAL_STEP', 'INTERVAL_LINEAR', 'INTERVAL_TYPES']

INTERVAL_CONSTANT = 'constant'
INTERVAL_STEP = 'step'
INTERVAL_LINEAR = 'linear'
INTERVAL_TYPES = [INTERVAL_CONSTANT, INTERVAL_STEP, INTERVAL_LINEAR]

_default_interval = 1
_invalid_value = -1


@dataclass
class PreconditioningConfig:
    num_total_steps: int = None
    preconditioner_upd_interval: int = _default_interval
    preconditioner_warmup_steps: int = 0
    preconditioner_upd_ratio: float = 1.
    preconditioner_warmup_ratio: float = 0.
    preconditioner_interval_type = INTERVAL_CONSTANT
    curvature_upd_interval: int = None
    curvature_warmup_steps: int = 0
    curvature_upd_ratio: float = None
    curvature_warmup_ratio: float = 0.
    curvature_interval_type = INTERVAL_CONSTANT
    data_size: int = _invalid_value
    damping: float = 1.e-7
    ema_decay: float = _invalid_value
    ignore_modules: List[Any] = None
    precondition_modules: List[Any] = None
    optimizer: torch.optim = None
    grafting: str = None
    norm_shrink: bool = False

class PreconditionedGradientMaker(GradientMaker):
    _supported_classes = None

    def __init__(self, model: nn.Module, config: PreconditioningConfig):
        super().__init__(model)
        self.config = config

        self.curvature_upd_schedule = None
        if config.num_total_steps is None:
            self.preconditioner_upd_schedule = None
        else:
            if config.preconditioner_upd_interval != _default_interval:
                warnings.warn('Since num_total_steps is specified, preconditioner_upd_interval '
                              'is ignored and preconditioner_upd_ratio is used.')
            self.preconditioner_upd_schedule = get_update_schedule(num_total_steps=config.num_total_steps,
                                                                   update_ratio=config.preconditioner_upd_ratio,
                                                                   warmup_ratio=config.preconditioner_warmup_ratio)
            if config.curvature_upd_ratio is not None:
                if config.curvature_upd_interval is not None:
                    warnings.warn('Since num_total_steps is specified, curvature_upd_interval '
                                  'is ignored and curvature_upd_ratio is used.')
                self.curvature_upd_schedule = get_update_schedule(num_total_steps=config.num_total_steps,
                                                                  update_ratio=config.curvature_upd_ratio,
                                                                  warmup_ratio=config.curvature_warmup_ratio)
        self.state = dict(step=0)
        self.module_dict = nn.ModuleDict({name.replace('.', '_'): m for name, m in model.named_modules()
                                          if self._is_supported(name, m)})
        self.module_inverse_dict = inverse_dict(self.module_dict)
        self.device = next(self.module_dict.parameters()).device
        print(self.module_dict)

    def add_ignore_module(self,ignore_modules=None,precondition_modules=None):
        if ignore_modules is not None:
            self.config.ignore_modules = ignore_modules
        if precondition_modules is not None:
            self.config.precondition_modules = precondition_modules
        self.module_dict = nn.ModuleDict({name.replace('.', '_'): m for name, m in self.model.named_modules()
                                          if self._is_supported(name, m)})
        print(self.module_dict)

    def _is_supported(self, module_name: str, module: nn.Module) -> bool:
        if len(list(module.children())) > 0:
            return False
        if all(not p.requires_grad for p in module.parameters()):
            return False
        ignore_modules = self.config.ignore_modules
        precond_modules = self.config.precondition_modules

        if ignore_modules is not None:
            for ignore_module in ignore_modules:
                if isinstance(ignore_module, type):
                    if isinstance(module, ignore_module):
                        return False
                elif isinstance(ignore_module, str):
                    if ignore_module in module_name:
                        return False
                elif ignore_module is module:
                    return False
        if self._supported_classes is not None:
            if not isinstance(module, self._supported_classes):
                warnings.warn(f'This model contains {module}, but ASDL library does not support {module}.')
                return False
            
        if precond_modules is not None:
            for precond_module in precond_modules:
                if isinstance(precond_module, type):
                    if isinstance(module, precond_module):
                        return True
                elif isinstance(precond_module, str):
                    if precond_module in module_name:
                        return True
                elif precond_module is module:
                    return True
            self.config.ignore_modules.append(module_name)
            return False
        return True

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state_dict: dict):
        self.state['step'] = state_dict['step']

    def forward_and_backward(self):
        step = self.state['step']

        self._startup()

        if self.do_forward_and_backward(step):
            self.forward()
            self.backward()
        if self.do_update_curvature(step):
            self.update_curvature()
        if self.do_update_preconditioner(step):
            self.update_preconditioner()

        if self.config.grafting is not None:
            self.save_grad()

        if self.config.norm_shrink:
            self.save_norm()

        self.precondition()

        if self.config.norm_shrink:
            self.adjust_norm()

        if self.config.grafting is not None:
            self.graft()

        self._teardown()

        self.state['step'] += 1

        return self._model_output, self._loss

    def _startup(self):
        pass

    def update_curvature(self):
        pass

    def update_preconditioner(self):
        pass

    def precondition(self):
        pass

    def _teardown(self):
        pass

    def do_forward_and_backward(self, step=None) -> bool:
        #raise NotImplementedError
        return True

    def do_update_curvature(self, step=None) -> bool:
        if self.curvature_upd_schedule is not None:
            return self._do_update_by_schedule(self.curvature_upd_schedule, step)
        config = self.config
        if config.curvature_upd_interval is not None:
            interval, warmup_steps = config.curvature_upd_interval, config.curvature_warmup_steps
            return self._do_update_by_interval(interval, warmup_steps, step)
        return self.do_update_preconditioner(step)

    def do_update_preconditioner(self, step=None) -> bool:
        if self.preconditioner_upd_schedule is not None:
            return self._do_update_by_schedule(self.preconditioner_upd_schedule, step)
        interval, warmup_steps = self.config.preconditioner_upd_interval, self.config.preconditioner_warmup_steps
        return self._do_update_by_interval(interval, warmup_steps, step)

    def _do_update_by_schedule(self, upd_schedule, step=None) -> bool:
        if step is None:
            step = self.state['step']
        try:
            return upd_schedule[step]
        except IndexError:
            warnings.warn(f'Given step+1 ({step+1}) is larger than the total number of steps ({self.config.num_total_steps})')
            return False

    def _do_update_by_interval(self, interval, warmup_steps=0, step=None) -> bool:
        if step is None:
            step = self.state['step']
        return step < warmup_steps or (step - warmup_steps) % interval == 0

    def graft(self):
        if self.config.grafting==None or self.config.grafting=='AllSGD':
            return
        for name, module in self.module_dict.items():
            if module.weight is None:
                continue
            if module.weight.prev_grad is not None:
                self.graft_module(module.weight)
            if module.bias is None:
                continue
            if module.bias.prev_grad is not None:
                self.graft_module(module.bias)

    def graft_module(self,module_weight):
        prev_grad = getattr(module_weight,'prev_grad')
        grad = getattr(module_weight,'grad')
        graft_norm = torch.norm(prev_grad)
        soo_norm = torch.norm(grad)
        if self.config.grafting=='SGDNorm':
            grad.mul_(graft_norm / (soo_norm + 1e-16))
        elif self.config.grafting=='SGDDirection':
            prev_grad.mul_(soo_norm / (graft_norm+ 1e-16))
            grad = prev_grad
        setattr(module_weight,'grad',grad)
        del prev_grad,grad

    def save_norm(self):
        for name, module in self.module_dict.items():
            if module.weight is None:
                continue
            if module.weight.grad is not None:
                module.weight.norm=torch.norm(module.weight.grad)
            if module.bias is not None and module.bias.grad is not None:
                module.bias.norm=torch.norm(module.bias.grad)

    def adjust_norm(self):
        for name, module in self.module_dict.items():
            if module.weight is None:
                continue
            if module.weight.grad is not None:
                if torch.norm(module.weight.grad) > module.weight.norm:
                    module.weight.grad = (module.weight.norm/torch.norm(module.weight.grad))*module.weight.grad
            if module.bias is not None and module.bias.grad is not None:
                if torch.norm(module.bias.grad) > module.bias.norm:
                    module.bias.grad = (module.bias.norm/torch.norm(module.bias.grad))*module.bias.grad

    def add_damping(self):
        return

    def remove_damping(self):
        return

def get_update_schedule(num_total_steps: int,
                        update_ratio: float = 1.,
                        warmup_ratio: float = 0.,
                        interval_type: str = INTERVAL_CONSTANT,
                        reverse=False):
    if num_total_steps <= 0:
        raise ValueError(f'num_total_steps has to be > 0. Got {num_total_steps}.')
    if update_ratio < 0 or 1 < update_ratio:
        raise ValueError(f'update_ratio has to be in [0, 1]. Got {update_ratio}.')
    num_total_updates = int(num_total_steps * update_ratio)
    if warmup_ratio < 0 or 1 < warmup_ratio:
        raise ValueError(f'warmup_ratio has to be in [0, 1]. Got {warmup_ratio}.')
    num_warmup_steps = int(num_total_steps * warmup_ratio)
    if num_warmup_steps > num_total_updates:
        raise ValueError(f'num_warmup_steps cannot be larger than num_total_updates ({num_total_updates}). '
                         f'Got {num_warmup_steps}.')
    if interval_type not in INTERVAL_TYPES:
        raise ValueError(f'Invalid interval_type: {interval_type}. {INTERVAL_TYPES} are supported.')

    update_schedule = [True] * num_warmup_steps
    num_remaining_steps = num_total_steps - num_warmup_steps
    num_remaining_updates = num_total_updates - num_warmup_steps

    if num_remaining_updates > 0:
        if interval_type == INTERVAL_CONSTANT:
            # constant interval
            interval = math.floor(num_remaining_steps / num_remaining_updates)
            for i in range(num_remaining_steps):
                update_schedule.append(i % interval == 0 and update_schedule.count(True) < num_total_updates)
        elif interval_type == INTERVAL_STEP:
            # step interval (one step)
            update_schedule.extend([True] * num_remaining_updates)
        else:
            # linear interval
            n = num_remaining_updates - 1
            interval = 1
            d_interval = math.floor(2 * (num_remaining_steps - n) / n / (n - 1))
            update_schedule.append(True)
            for i in range(n):
                update_schedule.extend([False] * (interval + d_interval * i - 1))
                update_schedule.append(True)

    if len(update_schedule) > num_total_steps:
        raise ValueError(f'len(update_schedule) cannot be larger than num_total_steps ({num_total_steps}). '
                         f'Got {len(update_schedule)}.')
    # padding with False
    update_schedule.extend([False] * (num_total_steps - len(update_schedule)))

    if reverse:
        return update_schedule[::-1]
    return update_schedule

def inverse_dict(d):
    return {v:k for k,v in d.items()}

def parameters_to_vector(parameters: Iterable[Tensor]) -> Tensor:
    # torch.nn.utils.parameters_to_vector uses param.view(-1) which doesn't work
    # with non-contiguous parameters
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def to_vector(x):
    return nn.utils.parameters_to_vector(x)

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))