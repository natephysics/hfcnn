import hydra
from hfcnn.utils import instantiate_list
from omegaconf import DictConfig, OmegaConf
import omegaconf
import os
OmegaConf.register_new_resolver("dict", lambda x, y: x.get(y))

@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(cfg: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from hfcnn.utils import disable_warnings, seed_everything

    if cfg.ignore_warnings:
        disable_warnings()

    if cfg.data.seed is not None:
        seed_everything(cfg.data.seed)

    if cfg.orig_wd is not None:
        os.environ['OWD'] = cfg.orig_wd

    #  Start action
    # return instantiate_list(cfg.action, cfg)
    return hydra.utils.instantiate(cfg.action, cfg)

if __name__ == "__main__":
    main()
