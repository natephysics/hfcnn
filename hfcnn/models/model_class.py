from typing import Optional
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from hfcnn.utils import instantiate_list
from torch import Tensor


class ImageClassificationBase(pl.LightningModule):
    """[summary]

    Args:
        pl ([type]): [description]
    """
    def __init__(
        self,
        criterion: Optional[DictConfig] = {},
        optimizer: Optional[DictConfig] = {},
        metrics: Optional[DictConfig] = {}
        ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "input_features", "criterion", "optimizer", "metrics"
        )
        self.network = None
        self.criterion = instantiate(criterion)
        self.optimizer = optimizer

        metrics = instantiate_list(metrics)
        metrics = MetricCollection(metrics)
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, *args, **kwargs) -> Tensor:
        for layer in self.network:
            x = layer(x)
        return x

    def step(self, batch: any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: any, batch_idx: int):
        loss, _, y = self.step(batch, batch_idx)
        self.log("train/loss", loss)
        if batch_idx != 0:
            for layer, param in self.named_parameters():
                self.log(f"train/{layer}.max_grad", torch.max(param.grad))
        return loss

    def validation_step(self, batch: any, batch_idx: int):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("val/loss", loss)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics)
        return loss

    def test_step(self, batch: any, batch_idx: int):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("test/loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)
        return loss

    def predict_step(self, batch: any, batch_idx: int):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters())
        return optimizer
    
    def _forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
