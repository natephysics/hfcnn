# %%
import os
import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List
from omegaconf import DictConfig
from hfcnn.utils import get_logg
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import LightningLoggerBase


# %%

