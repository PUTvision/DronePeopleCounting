import torch
from torchmetrics import Metric
import monai.metrics


class DiceMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        super(DiceMetric, self).__init__()

        self.metric = monai.metrics.DiceMetric(
            include_background=False,
        )

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        assert inputs.shape == targets.shape

        self.metric(y_pred=inputs, y=targets.type(torch.int))

    def compute(self):
        return self.metric.aggregate().item()

    def reset(self):
        self.metric.reset()
