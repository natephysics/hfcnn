from typing import Optional, List
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from hfcnn.utils import instantiate_list
from torch import Tensor
import torch


class ImageClassificationBase(pl.LightningModule):
    """[summary]"""

    def __init__(
        self,
        log_training: Optional[bool] = False,
        criterion: Optional[DictConfig] = {},
        optimizer: Optional[DictConfig] = {},
        scheduler: Optional[DictConfig] = {},
        metrics: Optional[DictConfig] = {},
        input_dim: int or List[int] = None,
        output_dim: int or List[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "log_training", "criterion", "optimizer", "scheduler", "metrics"
        )
        self.network = None
        self.criterion = instantiate(criterion)
        self.optimizer = optimizer
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Metrics
        metrics = instantiate_list(metrics)
        metrics = MetricCollection(metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # define "lr" as attribute to leverage pl learning rate tuner
        self.optimizer = optimizer
        self.lr = optimizer.lr

        #  Whether to log gradient and weights during training
        self.log_training = log_training

        self.scheduler = scheduler

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for layer in self.network:
            x = layer(x)
        return x

    def step(self, batch: any, batch_idx: int):
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        if torch.isnan(loss):
            print("Issue")
        return loss, y_hat, y

    def training_step(self, batch: any, batch_idx: int):
        loss, y_hat, y = self.step(batch, batch_idx)
        # self.log("train/loss", loss)
        self.train_metrics(y_hat, y)
        if self.log_training:
            self.log_dict(self.train_metrics)
            for layer, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    f"train/{layer}", param, global_step=self.global_step
                )
                if batch_idx != 0:
                    self.log(f"train/{layer}.max_grad", torch.max(param.grad))
        del y_hat, y
        return loss

    def validation_step(self, batch: any, batch_idx: int):
        loss, y_hat, y = self.step(batch, batch_idx)
        # self.log("val/loss", loss)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics)
        del y_hat, y
        return loss

    def test_step(self, batch: any, batch_idx: int):
        loss, y_hat, y = self.step(batch, batch_idx)
        # self.log("test/loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)
        del y_hat, y
        return loss

    def predict_step(self, batch: any, batch_idx: int):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters(), lr=self.lr)
        if self.scheduler != {}:
            scheduler = self.scheduler.copy()
            monitor = scheduler.pop("monitor", None)
            lr_scheduler = {
                "scheduler": instantiate(scheduler, optimizer),
                "monitor": monitor,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer
