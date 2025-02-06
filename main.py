import hydra
from omegaconf import OmegaConf

from src.config import Config


def func(x: float):
    return (x-2.5)**2


@hydra.main(config_path="configs", config_name="config", version_base='1.2')
def main(cfg: Config):
    print(cfg)
    cfg: Config = OmegaConf.to_object(cfg)
    cfg.initialize()
    return func(cfg.train.lr)


if __name__ == "__main__":
    main()
