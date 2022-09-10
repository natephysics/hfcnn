import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import LightningLoggerBase
from numpy import set_printoptions

# for exporting the data
set_printoptions(linewidth=100000)

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

OmegaConf.register_new_resolver("dict", lambda x, y: x.get(y))


@hydra.main(version_base="1.2", config_path="configs/", config_name="prepare.yaml")
def main(cfg: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from hfcnn import utils
    import os
    from hfcnn.preprocessing import prepare_data, prepare_test_data
    from hfcnn.datamodules.heat_load_data import HeatLoadDataModule

    if cfg.ignore_warnings:
        utils.disable_warnings()

    if cfg.data.seed is not None:
        utils.seed_everything(cfg.data.seed)

    if cfg.orig_wd is not None:
        os.environ["OWD"] = cfg.orig_wd

    # Applies optional utilities
    utils.extras(cfg)

    loggers: LightningLoggerBase = utils.instantiate_list(cfg.logger)

    if cfg.prepare_test == True:
        prepare_test_data(cfg)

    return prepare_data(cfg)

    #  Start action (or list of actions)


if __name__ == "__main__":
    main()
