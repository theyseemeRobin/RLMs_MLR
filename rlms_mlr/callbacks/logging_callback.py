from typing import Dict, Optional
import os

from rlms_mlr.callbacks.base_callback import Callback, TrainerState, CallbackReturn
from rlms_mlr.loggers.base_logger import Logger


class LoggerCallback(Callback):
    """
    Logs hyperparameters, metrics, and saves models via the provided Logger.

    Args:
        logger: An implementation of the Logger interface.
        params: Hyperparameters dict to log once at train start.
        save_best_only: If True, only save when eval_loss improves.
    """

    def __init__(
            self,
            logger: Logger,
            params: Dict[str, any],
    ):
        self.logger = logger
        self.params = params

    def on_train_start(self, trainer_state: TrainerState, **kwargs) -> None:
        self.logger.log_config(self.params)

    def on_epoch_end(self, trainer_state: TrainerState, **kwargs) -> None:
        for name, value in trainer_state.logs.items():
            self.logger.log_metric(name, value, step=trainer_state.current_epoch)

        # TODO: fix model saving
        # eval_loss = trainer_state.logs.eval_loss
        # if eval_loss is not None:
        #     if not self.save_best_only or eval_loss < self.best_loss:
        #         self.best_loss = eval_loss
        #         os.makedirs(self.checkpoint_dir, exist_ok=True)
        #         ckpt_path = os.path.join(self.checkpoint_dir, f'best_epoch_{trainer_state.current_epoch}.pt')
        #         self.logger.save_model(trainer_state.model, ckpt_path)

    def on_train_end(self, trainer_state: TrainerState, **kwargs) -> None:
        self.logger.close()
