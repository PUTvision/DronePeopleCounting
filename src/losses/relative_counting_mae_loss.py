import torch
from torch import nn


class RelativeCountingMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        tensor_dims = len(inputs.size())
        dim = tuple(range(1, tensor_dims))

        preds_people = torch.sum(inputs, dim=dim) / 400
        target_people = torch.sum(targets, dim=dim) / 400

        return torch.mean(torch.div(torch.abs(preds_people - target_people), target_people+1))
