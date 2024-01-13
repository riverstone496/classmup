import torch
import torch.distributed as dist

class Metric(object):
    def __init__(self, device):
        self._n = torch.tensor([0.0]).to(device)
        self._loss = torch.tensor([0.0]).to(device)
        self._corrects = torch.tensor([0.0]).to(device)
        self._corrects_5 = torch.tensor([0.0]).to(device)

    def update(self, n, loss, outputs, targets):
        with torch.inference_mode():
            self._n += n
            self._loss += loss * n
            _, preds = torch.max(outputs, 1)
            self._corrects += torch.sum(preds == targets)

    def sync(self):
        dist.all_reduce(self._n, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._corrects, op=dist.ReduceOp.SUM)

    @property
    def loss(self):
        return (self._loss / self._n).item()

    @property
    def accuracy(self):
        return (self._corrects / self._n).item()

    def __str__(self):
        return f'Loss: {self.loss:.4f}, Acc: {self.accuracy:.4f}'
