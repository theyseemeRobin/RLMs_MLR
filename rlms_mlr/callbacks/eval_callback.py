from typing import Iterator

import torch

from rlms_mlr.callbacks.base_callback import Callback, TrainerState, CallbackReturn, Logs
from rlms_mlr.data.dataset import Batch


class EvalCallback(Callback):

    def __init__(self, eval_loader: Iterator[Batch]):
        self.eval_loader = eval_loader

    def on_epoch_end(self, trainer_state: TrainerState, **kwargs) -> None:
        """
        Evaluate the model on the validation set at the end of each epoch.
        """
        trainer_state.model.eval()

        mean_metrics = {}
        trainer_state.model.eval()
        for batch_idx, batch in enumerate(self.eval_loader):
            batch = batch.to(trainer_state.model.device, non_blocking=True)
            with torch.no_grad():
                metrics = trainer_state.model.evaluate(batch)
            for key, value in metrics.items():
                if key not in mean_metrics:
                    mean_metrics[key] = 0
                mean_metrics[key] += (value - mean_metrics[key]) / (batch_idx + 1)

        trainer_state.logs = Logs(**mean_metrics)