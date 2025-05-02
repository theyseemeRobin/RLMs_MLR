import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore

from rlms_mlr.trainer import TrainerConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: TrainerConfig):

    # retrieve the hydra run dir
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory: {out_dir}")

    trainer_cfg: TrainerConfig = instantiate(cfg)
    trainer = trainer_cfg.get_trainer()
    trainer.train()

if __name__ == "__main__":
    main()