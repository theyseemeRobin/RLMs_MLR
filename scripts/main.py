import hydra
from hydra.utils import instantiate
import optuna
from omegaconf import OmegaConf, DictConfig

from src.utils.hp_sampling import sample_hp


def start(cfg: DictConfig, trial: optuna.Trial = None):

    # retrieve the hydra run dir
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory: {out_dir}")

    # sample hyperparameters
    if trial is not None:
        sampled_hp = sample_hp(trial, cfg.tuning.hyperparameters)
        for key, value in sampled_hp.items():
            OmegaConf.update(cfg, key, value)
    return 0


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg):

    # setup optuna HP search
    if cfg.get('tuning') is not None:
        study = optuna.create_study(
            study_name=cfg.tuning.study_name,
            storage=cfg.tuning.storage,
            load_if_exists=True,
            direction=cfg.tuning.direction,
            pruner=instantiate(cfg.tuning.pruner)
        )
        study.optimize(
            lambda trial: start(cfg, trial),
            n_trials=cfg.tuning.n_trials
        )
    else:
        start(cfg)


if __name__ == "__main__":

    main()