import torch
from torch import nn


class CountingMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        tensor_dims = len(inputs.size())
        dim = tuple(range(1, tensor_dims))

        return torch.mean(torch.abs(
            torch.sum(inputs, dim=dim) / 400 - torch.sum(targets, dim=dim) / 400))
