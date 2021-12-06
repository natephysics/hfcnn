import os
import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List
from omegaconf import DictConfig
from hfcnn.utils import get_logger, instantiate_list, build_default_paths
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import LightningLoggerBase

def train(cfg: DictConfig, **kwargs) -> None:

    log = get_logger(__name__)

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
    if cfg.trainer.gpus is not None and cfg.trainer.gpus != 0:
        pin_memory = True

    #  Instatiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule,
        train_data_path=default_paths['train'],
        val_data_path=default_paths['validation'],
        test_data_path=default_paths['test'],
        data_root=default_paths['raw_folder'],
        params_file_path=params_file_path,
        pin_memory=pin_memory,
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

    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        criterion=cfg.criterion,
        optimizer=cfg.optimizer,
        metrics=cfg.metric,
        input_dim=input_dim,
        output_dim=output_dim,
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

    # Instantiate logger
    loggers: LightningLoggerBase = instantiate_list(cfg.logger)

    deterministic = True if cfg.data.seed is not None else False

    if not isinstance(loggers, list):
        loggers = [loggers]

    # Instantate trainer 
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=loggers, callbacks=callbacks, deterministic=deterministic
    )
    
    for i, logger in enumerate(loggers):
        log.info(f"Logger {i}: {logger.__class__.__name__} instantiated.")

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
    hparams["seed"] = cfg.data.seed

    #  Add datamodule metrics
    hparams["datamodule/num_train"] = len(datamodule.train_data)
    hparams["datamodule/num_val"] = len(datamodule.val_data)
    hparams["datamodule/num_test"] = len(datamodule.test_data)

    #  Add hp metrics
    # hp_metrics = {}
    # hp_metrics["val/loss"] = 0

    #  Log hparams
    #  TensorBoard requires metrics to be defined with hyperparameters
    for logger in loggers:
        logger.log_hyperparams(hparams)
        # if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
        #     logger.log_hyperparams(hparams, hp_metrics)
        # else:
        #     logger.log_hyperparams(hparams)
        #     logger.log_metrics(hp_metrics)

    # disable logging any more hyperparameters for all loggers
    # see: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
    def empty(*args, **kwargs):
        pass
    
    for logger in loggers:
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

    # retreive best validation loss from training
    best_model_score = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            if callback.best_model_score is not None:
                best_model_score = callback.best_model_score.item()
            break
    
    ########
    # Test #
    ########

    if cfg.data.test_strategy is None:
        for logger in loggers: 
            logger.finalize(status="FINISHED")
        return

    #  Test model on supported strategies:
    #  - None: exit
    #  - test: use test data from datamodule
    #  - train: use train data from datamodule

    log.info("Test the model according to test strategy: %s" % cfg.data.test_strategy)

    if cfg.data.test_strategy == "test":
        dataset = datamodule.test_data
    elif cfg.data.test_strategy == "train":
        dataset = datamodule.train_data
    else:
        raise ValueError("%s test strategy is not supported" % cfg.data.test_strategy)

    #  Test first data with trainer
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=bool(cfg.datamodule.num_workers),
        num_workers=cfg.datamodule.num_workers
    )
    trainer.test(model, dataloaders=dataloader)

    ############
    # Finalize #
    ############

    #  Finalize objects
    for logger in loggers:
        logger.finalize(status="FINISHED")

    return best_model_score