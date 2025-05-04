from typing import Iterator
import torch

from rlms_mlr.callbacks.base_callback import Callback, TrainerState
from rlms_mlr.data.dataset import Batch
from rlms_mlr.models.base_model import Metrics


class EvalCallback(Callback):

    def __init__(self, eval_loader: Iterator[Batch]):
        self.eval_loader = eval_loader

    def on_epoch_end(self, trainer_state: TrainerState, **kwargs) -> None:
        """
        Evaluate the model on the validation set at the end of each epoch.
        """
        trainer_state.model.eval()

        trainer_state.model.eval()
        metrics = None
        for batch_idx, batch in enumerate(self.eval_loader):
            batch = batch.to(trainer_state.model.device, non_blocking=True)
            with torch.no_grad():
                new_metrics = trainer_state.model.evaluate(batch)

            if metrics is None:
                metrics = new_metrics
            else:
                metrics.merge(new_metrics)

        trainer_state.logs.update(metrics.compute_metrics())