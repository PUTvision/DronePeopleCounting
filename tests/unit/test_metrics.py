import pytest

import torch

from src.metrics.counting_mae_metric import CountingMAEMetric
from src.metrics.dice_metric import DiceMetric


def test_PeopleMAE():
    metric = CountingMAEMetric()
    zeros = torch.zeros((4, 1, 32, 32))
    dest = torch.zeros((4, 1, 32, 32))
    dest[:, 0, 1, :5] = 100
    metric.update(dest, zeros)
    assert torch.isclose(metric.compute(), torch.tensor(5.0))
    del metric

    metric = CountingMAEMetric()
    zeros = torch.zeros((4, 1, 32, 32))
    dest = torch.zeros((4, 1, 32, 32))
    dest[0, 0, 1, :2] = 100
    metric.update(dest, zeros)
    assert torch.isclose(metric.compute(), torch.tensor(0.5))

    dest[1, 0, 1, 10:] = 100
    metric.update(dest, zeros)
    assert torch.isclose(metric.compute(), torch.tensor(3.25))

