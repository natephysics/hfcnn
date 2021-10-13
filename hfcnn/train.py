import os
import torch
from  shutil import copyfile
import hydra
import pytorch_lightning as pl
from typing import List
from omegaconf import DictConfig
from hfcnn.utils import get_logger, instantiate_list
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import LightningLoggerBase

def train(cfg: DictConfig, **kwargs) -> None:

    log = get_logger(__name__)

    ##############
    # Datamodule #
    ##############

    processed_data_root = os.path.join(cfg.orig_wd, cfg.data_folder)
    
    ## TODO: add param files
    if cfg.datamodule.params_file_path:
        params_file_path = os.path.join(cfg.orig_wd, cfg.datamodule.params_file_path)
    else:
        params_file_path = False

    num_workers = os.cpu_count()
    pin_memory = False
    if cfg.trainer.gpus is not None and cfg.trainer.gpus != 0:
        num_workers = 1
        pin_memory = True

    #  Instatiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule,
        data_root=processed_data_root,
        params_file_path=params_file_path,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    # Setup the data set
    datamodule.setup()

    # Save a copy of the data in the wd
    datamodule.save_data(cfg.data_folder)

    log.info("DataModule: %s" % datamodule)


    #########
    # Model #
    #########

    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        criterion=cfg.criterion,
        optimizer=cfg.optimizer,
        metrics=cfg.metric,
        _recursive_=False, # for hydra (won't recursively instantiate criterion)
    )
    log.info("Model: %s" % model)

    # if cfg.zeros_weight:
    #     model.zeros_all_trainable_parameters()

    ###########
    # Trainer #
    ###########

    callbacks: List[Callback] = instantiate_list(cfg.callback)
    for c in callbacks:
        log.info("Callback: %s" % c.__class__.__name__)

    #  Instantiate logger
    logger: LightningLoggerBase = hydra.utils.instantiate(cfg.logger)
    log.info("Logger: %s" % logger.__class__.__name__)

    deterministic = True if cfg.seed is not None else False
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks, deterministic=deterministic
    )
    log.info("Trainer: %s" % trainer.__class__.__name__)

    ###################
    # Hyperparameters #
    ###################

    hparams = {}

    hparams["datamodule"] = cfg.datamodule
    hparams["model"] = cfg.model
    hparams["optimizer"] = cfg.optimizer
    hparams["criterion"] = cfg.criterion
    hparams["trainer"] = cfg.trainer

    if "callback" in cfg:
        hparams["callback"] = cfg.callback

    #  Add additional training hps
    hparams["seed"] = cfg.seed

    #  Add datamodule metrics
    hparams["datamodule/num_train"] = len(datamodule.train_data)
    hparams["datamodule/num_val"] = len(datamodule.val_data)
    hparams["datamodule/num_test"] = len(datamodule.test_data)

    #  Add hp metrics
    hp_metrics = {}
    hp_metrics["val/loss"] = 0

    #  Log hparams
    #  TensorBoard requires metrics to be defined with hyperparameters
    if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
        pass
        # logger.log_hyperparams(hparams, hp_metrics)
    else:
        logger.log_hyperparams(hparams)
        logger.log_metrics(hp_metrics)

    # disable logging any more hyperparameters for all loggers
    # see: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
    def empty(*args, **kwargs):
        pass

    log_hyperparams_ = logger.log_hyperparams
    logger.log_hyperparams = empty

    #  Log hparams to console too
    log.info("Num of train samples: %d" % len(datamodule.train_data))
    log.info("Num of validation samples: %d" % len(datamodule.val_data))
    log.info("Num of test samples: %d" % len(datamodule.test_data))
    log.info("Train transforms: %s" % datamodule.train_data.settings['transforms'])




    #########
    # Train #
    #########

    log.info("Check loss before training ...")
    trainer.validate(model, datamodule=datamodule)

    log.info("Start training ...")
    trainer.fit(model, datamodule=datamodule)

    ########
    # Test #
    ########

    if cfg.test_strategy is None:
        logger.finalize(status="FINISHED")
        return

    #  Test model on supported strategies:
    #  - None: exit
    #  - test: use test data from datamodule
    #  - train: use train data from datamodule

    log.info("Test the model according to test strategy: %s" % cfg.test_strategy)

    if cfg.test_strategy == "test":
        dataset = datamodule.test_data
    elif cfg.test_strategy == "train":
        dataset = datamodule.train_data
    else:
        raise ValueError("%s test strategy is not supported" % cfg.test_strategy)

    
