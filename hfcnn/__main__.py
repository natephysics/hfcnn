import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(cfg: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from hfcnn.utils import disable_warnings, seed_everything

    if cfg.ignore_warnings:
        disable_warnings()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    #  Start action
    return hydra.utils.instantiate(cfg.action, cfg)


if __name__ == "__main__":
    main()
