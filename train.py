import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

OmegaConf.register_new_resolver("dict", lambda x, y: x.get(y))

@hydra.main(config_path="configs/", config_name="train.yaml")
def main(cfg: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from hfcnn import utils
    import os

    if cfg.ignore_warnings:
        utils.disable_warnings()

    if cfg.data.seed is not None:
        utils.seed_everything(cfg.data.seed)

    if cfg.orig_wd is not None:
        os.environ['OWD'] = cfg.orig_wd

    # Applies optional utilities
    utils.extras(cfg)

    #  Start action (or list of actions)
    if '_target_' in cfg.action.keys():
        return hydra.utils.instantiate(cfg.action, cfg)
    else: 
        return utils.instantiate_list(cfg.action, cfg)

if __name__ == "__main__":
    main()