from typing import Optional

from rlms_mlr.callbacks.base_callback import Callback, TrainerState, CallbackReturn


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback to stop training when the validation loss does not improve for a given number of epochs.
    Args:
        patience: Number of epochs with no improvement after which training will be stopped.
    """

    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        loss = trainer_state.logs.eval_loss
        if loss >= self.best:
            self.wait += 1
            if self.wait >= self.patience:
                return CallbackReturn(stop_training=True)
        else:
            self.best = loss
            self.wait = 0
        return None