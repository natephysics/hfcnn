import os
import logging
import warnings
import torch
from typing import List
from hydra.utils import instantiate
from pytorch_lightning import seed_everything as seed_everthing
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """
    Initializes multi-GPU-friendly python logger.

    From: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    seed_everthing(seed, workers=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def disable_warnings() -> None:
    """Disable python warnings."""
    warnings.filterwarnings("ignore")
    log = get_logger(__name__)
    log.info("Python warnings are disabled")


def instantiate_list(cfg: DictConfig, **kwargs) -> List[any]:
    """Instatiate a list through hydra instantiate."""
    objects: List[any] = []
    for _, cfg_ in cfg.items():
        if "_target_" in cfg_:
            #  Add kwargs if found under configuration
            kwargs_ = {k: v for k, v in kwargs.items() if k in cfg_}
            objects.append(instantiate(cfg_, **kwargs_))
    return objects
