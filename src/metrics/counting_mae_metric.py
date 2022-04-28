import torch
import torch.nn.functional
from torchmetrics import Metric


class CountingMAEMetric(Metric):
    def __init__(self):
        super().__init__()

        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        tensor_dims = len(inputs.size())
        dim = tuple(range(1, tensor_dims))

        self.sum += torch.sum(torch.abs(torch.sum(inputs, dim=dim) / 100 - torch.sum(targets, dim=dim) / 100))
        self.count += inputs.size(dim=0)

    def compute(self):
        return self.sum / self.count
