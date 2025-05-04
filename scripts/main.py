import logging

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from rlms_mlr.callbacks.logging_callback import LoggerCallback
from rlms_mlr.loggers.tensorboard_logger import TensorBoardLogger
from rlms_mlr.loggers.plotext_logger import PlotextLogger
from rlms_mlr.trainer import Trainer


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    # retrieve the hydra run dir
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(f"Output directory: {out_dir}")

    trainer: Trainer = instantiate(cfg.trainer)
    trainer.add_callback(LoggerCallback(TensorBoardLogger(cfg.log_dir), OmegaConf.to_container(cfg)))
    trainer.add_callback(LoggerCallback(PlotextLogger(['train_loss', 'val_loss', 'val_accuracy']),
                                        OmegaConf.to_container(cfg)))
    trainer.train(**cfg.train_kwargs)

if __name__ == "__main__":
    main()