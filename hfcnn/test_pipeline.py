import os
import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchinfo import summary
from typing import List
from omegaconf import DictConfig
from hfcnn.utils import (
    get_logger,
    instantiate_list,
    build_default_paths,
    seed_everything,
)
from hfcnn.models.hf_model import GoogleNet
from hfcnn.datamodules.heat_load_data import HeatLoadDataModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import LightningLoggerBase


def test(cfg: DictConfig, **kwargs) -> None:

    log = get_logger(__name__)

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.data.get("seed"):
        seed_everything(cfg.data.seed)

    default_paths = build_default_paths(cfg)

    ##############
    # Datamodule #
    ##############

    ## TODO: add param files
    if cfg.datamodule.params_file_path:
        params_file_path = os.path.join(cfg.orig_wd, cfg.datamodule.params_file_path)
    else:
        params_file_path = False

    pin_memory = False

    #  Instatiate datamodule
    datamodule: HeatLoadDataModule = HeatLoadDataModule.load_from_checkpoint(
        cfg.ckpt_path,
        train_data_path=default_paths["train"],
        val_data_path=default_paths["validation"],
        test_data_path=default_paths["test"],
        data_root=default_paths["raw_folder"],
        params_file_path=params_file_path,
        pin_memory=pin_memory,
        cache=False,
        batch_size=1,
        num_workers=0,
    )

    # Setup the data set
    datamodule.setup()

    # Save a copy of the data in the hydra wd
    datamodule.save_data(cfg.data.save_data_folder)

    # Extract the input/output dimension of the data set
    input_dim = datamodule.get_input_dim()
    output_dim = datamodule.get_output_dim()

    log.info("DataModule: %s" % datamodule)

    #########
    # Model #
    #########

    # model: LightningModule = hydra.utils.instantiate(
    #     cfg.model,
    #     criterion=cfg.criterion,
    #     optimizer=cfg.optimizer,
    #     metrics=cfg.metric,
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    #     _recursive_=False,  # for hydra (won't recursively instantiate criterion)
    # )
    model = GoogleNet.load_from_checkpoint(
        cfg.ckpt_path,
        act_fn=cfg.model.act_fn,
        figures=cfg.figures,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    log.info("Model: %s" % model)

    # NCHW order
    # N represents the batch dimension
    # C represents the channel dimension
    # H represents the image height (number of rows)
    # and W represents the image width (number of columns)
    C, H, W = input_dim
    log.info(f"Input shape: (N, C, H, W) = (N, {C}, {H}, {W})")
    # summary(model, input_size=(datamodule.batch_size, C, H, W), depth=0)
    # if cfg.zeros_weight:
    #     model.zeros_all_trainable_parameters()

    ###########
    # Trainer #
    ###########

    callbacks: List[Callback] = instantiate_list(cfg.callback)
    for c in callbacks:
        log.info("Callback: %s" % c.__class__.__name__)

    # Instantiate logger
    loggers: LightningLoggerBase = instantiate_list(cfg.logger)

    deterministic = True if cfg.data.seed is not None else False

    if not isinstance(loggers, list):
        loggers = [loggers]

    # Instantate trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers,
        callbacks=callbacks,
        deterministic=deterministic,
    )

    for i, logger in enumerate(loggers):
        log.info(f"Logger {i}: {logger.__class__.__name__} instantiated.")

    log.info("Trainer: %s" % trainer.__class__.__name__)

    #  Log hparams to console too
    log.info("Num of train samples: %d" % len(datamodule.train_data))
    log.info("Num of validation samples: %d" % len(datamodule.val_data))
    log.info("Num of test samples: %d" % len(datamodule.test_data))
    log.info("Train transforms: %s" % datamodule.train_data.settings["transforms"])

    ########
    # Test #
    ########

    #  Test model on supported strategies:
    #  - None: exit
    #  - test: use test data from datamodule
    #  - val: use val data from datamodule

    log.info("Test the model according to test strategy: %s" % cfg.test_strategy)

    if cfg.test_strategy == "test":
        trainer.test(model, dataloaders=datamodule)
    elif cfg.test_strategy == "val":
        trainer.validate(model, dataloaders=datamodule)
    else:
        raise ValueError("%s test strategy is not supported" % cfg.test_strategy)

    ############
    # Finalize #
    ############

    #  Finalize objects
    for logger in loggers:
        logger.finalize(status="FINISHED")

    return
