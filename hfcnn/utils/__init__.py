import os
import logging
import warnings
import torch
import rich.tree
import pytorch_lightning as pl
from typing import List, Sequence
from hydra.utils import instantiate
from pytorch_lightning import seed_everything as seed_everthing
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig
from hfcnn.yaml_tools import import_configuration
from numpy import array
from re import findall


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

# log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        ju(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


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


def instantiate_list(cfg: DictConfig, *args, **kwargs) -> List[any]:
    """Instatiate a list through hydra instantiate."""
    objects: List[any] = []
    for _, cfg_ in cfg.items():
        if "_target_" in cfg_:
            #  Add kwargs if found under configuration
            kwargs_ = {k: v for k, v in kwargs.items() if k in cfg_}
            for key, value in cfg_.items():
                cfg_[key] = value
            objects.append(instantiate(cfg_, *args, **kwargs_))
    return objects


def build_default_paths(cfg: DictConfig) -> DictConfig:
    """Updates the cfg with the default paths

    Args:
        cfg (DictConfig): The config file generated by Hydra

    Returns:
        DictConfig: config with updated paths
    """
    path_to_default_config = "hfcnn/default_settings/default_paths.yaml"

    # improt the correct paths
    default_paths = import_configuration(os.path.join(cfg.orig_wd, path_to_default_config))

    for key in default_paths.keys():
        # use the default path if available
        if key in cfg.keys():
            default_paths[key] = os.path.join(cfg.orig_wd, cfg[key])
        # if exists, if the path isn't absolute, update the path to include the correct working directory 
        else:
            default_paths[key] = os.path.join(cfg.orig_wd, default_paths[key])
                

    return default_paths

def str_to_array(x): 
    """Takes in a string and returns an array of floats."""
    return array(findall(r'[\de\+\-\.]+', x), dtype=float)