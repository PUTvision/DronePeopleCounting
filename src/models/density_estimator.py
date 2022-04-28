from typing import Optional, Tuple, List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics
from neptune.new.types import File
from torch import nn

from src.losses.counting_mae_loss import CountingMAELoss
from src.losses.relative_counting_mae_loss import RelativeCountingMAELoss
from src.metrics.counting_mae_metric import CountingMAEMetric


class DensityEstimator(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 encoder_name: str,
                 input_channels: int,
                 classes: List[str],
                 loss_function: str,
                 lr: float,
                 lr_patience: int,
                 visualize_test_images: bool):
        super().__init__()

        self._model_name = model_name
        self._encoder_name = encoder_name
        self._input_channels = input_channels
        self._classes = classes
        self._loss_function = loss_function
        self._lr = lr
        self._lr_patience = lr_patience
        self._visualize_test_images = visualize_test_images

        if self._model_name == 'UNet':
            self.network = smp.Unet
        elif self._model_name == 'DeepLabV3Plus':
            self.network = smp.DeepLabV3Plus
        else:
            raise NotImplementedError(
                f'Unsupported model: {self._model_name}')

        self.network = self.network(
            encoder_name=self._encoder_name,
            encoder_weights="imagenet",
            in_channels=self._input_channels,
            classes=len(self._classes),
            activation='softmax'
        )

        if loss_function == 'MAE':
            self.loss = nn.L1Loss()
        elif loss_function == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_function == 'CountingMAE':
            self.loss = CountingMAELoss()
        elif loss_function == 'RelativeCountingMAE':
            self.loss = RelativeCountingMAELoss()
        else:
            raise NotImplementedError(
                f'Unsupported loss function: {loss_function}')

        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanSquaredError(),
            CountingMAEMetric()

        ])
        self.train_metrics = metrics.clone('train_')
        self.valid_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        if torch.isinf(loss):
            return None

        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log_dict(self.train_metrics(y_pred, y))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_metrics(y_pred, y))

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('test_loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log_dict(self.test_metrics(y_pred, y))

        if self._visualize_test_images:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                          patience=self._lr_patience, min_lr=1e-6,
                                                                          verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': reduce_lr_on_plateau,
            'monitor': 'train_loss_epoch'
        }
