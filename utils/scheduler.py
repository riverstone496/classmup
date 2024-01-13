import math
from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr

class PolynomialDampingDecay():
    """Polynomial learning rate decay until step reach to max_decay_step
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    def __init__(self, total_steps,initial_dmp=1e-1, end_dmp=1e-6, power=4.0,warmup_steps=0):
        if total_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.initial_dmp = initial_dmp
        self.end_dmp = end_dmp
        self.warmup_steps = warmup_steps
        self.power = power
        self.total_steps = total_steps
        self.current_step=0

    def get_current_value(self):
        if self.current_step > self.total_steps:
            damping = self.end_dmp
        elif self.current_step < self.warmup_steps:
            damping = self.initial_dmp
        else:
            if self.end_dmp < self.initial_dmp:
                damping = self.end_dmp + (self.initial_dmp - self.end_dmp) * ((self.total_steps - self.current_step)/(self.total_steps - self.warmup_steps)) ** self.power
            elif self.end_dmp == self.initial_dmp:
                damping = self.end_dmp
            else:
                damping =  self.end_dmp - (self.end_dmp - self.initial_dmp)  * ((self.total_steps - self.current_step)/(self.total_steps - self.warmup_steps)) ** self.power
        self.current_step += 1
        return damping

class PolynomialInvDampingDecay():
    """Polynomial learning rate decay until step reach to max_decay_step
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    def __init__(self, total_steps,initial_dmp=1e-1, end_dmp=1e-6, power=4.0,warmup_steps=0):
        if total_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.initial_dmp = initial_dmp
        self.end_dmp = end_dmp
        self.warmup_steps = warmup_steps
        self.power = power
        self.total_steps = total_steps
        self.current_step=0

    def get_current_value(self):
        if self.current_step > self.total_steps:
            damping = self.end_dmp
        elif self.current_step < self.warmup_steps:
            damping = self.initial_dmp
        else:
            if self.initial_dmp > self.end_dmp :
                damping = self.initial_dmp - (self.initial_dmp - self.end_dmp) * (( self.current_step- self.warmup_steps)/(self.total_steps - self.warmup_steps)) ** self.power
            elif self.end_dmp == self.initial_dmp:
                damping = self.end_dmp
            else:
                damping =  self.initial_dmp + (self.end_dmp - self.initial_dmp)  * (( self.current_step- self.warmup_steps)/(self.total_steps - self.warmup_steps)) ** self.power
        self.current_step += 1
        return damping

class ExponentialDampingDecay():
    def __init__(self,total_steps,initial_dmp=1e-1,end_dmp = 1e-3, alpha=None,warmup_steps=0,warmup_alpha=1):
        self.current_damping = initial_dmp
        self.warmup_steps = warmup_steps
        self.warmup_alpha =  warmup_alpha
        if alpha is None:
            self.alpha = (end_dmp / initial_dmp)**(1/total_steps)
        else:
            self.alpha = alpha
        self.current_step=0
        print(self.alpha)

    def get_current_value(self):
        dmp = self.current_damping
        if self.current_step < self.warmup_steps:
            self.current_damping *= self.warmup_alpha
        else:
            self.current_damping *= self.alpha
        return dmp

class EmaDampingScheduler():
    def __init__(self, init_damping, target_damping,total_epoch,alpha=None):
        self.init_damping = init_damping
        self.target_damping = target_damping
        self.warmup_epoch = total_epoch
        self.damping=self.init_damping
        
        self.alpha = alpha
        if alpha is None:
            if init_damping >= target_damping:
                self.alpha=2*math.log10(init_damping/target_damping)/total_epoch
            else:
                self.alpha=2*math.log10(target_damping/init_damping)/total_epoch
        print(self.alpha)

    def get_current_value(self):
        damping=self.damping
        next_damping=(1-self.alpha)*damping+self.alpha*self.target_damping
        self.damping=next_damping
        return damping
    
    def get_values(self,T_max):
        damp_list=[self.init_damping]
        for i in range(T_max-1):
            damping=damp_list[-1]
            damp_list.append((1-self.alpha)*damping+self.alpha*self.target_damping)
        return damp_list