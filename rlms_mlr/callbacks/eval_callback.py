from typing import Iterator
import torch

from rlms_mlr.callbacks.base_callback import Callback, TrainerState
from rlms_mlr.data.dataset import Batch
from rlms_mlr.models.base_model import Metrics, Model


def evaluate_model(model: Model, loader: Iterator[Batch]) -> Metrics:
    model.eval()
    metrics = None
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(model.device, non_blocking=True)
        with torch.no_grad():
            new_metrics = model.evaluate(batch)

        if metrics is None:
            metrics = new_metrics
        else:
            metrics.merge(new_metrics)
    return metrics

class ValidateCallback(Callback):
    def on_epoch_start(self, trainer_state: TrainerState, **kwargs) -> None:
        """
        Evaluate the model on the validation set at the end of each epoch.
        """
        loader = trainer_state.data_module.val_dataloader()
        metrics = evaluate_model(trainer_state.model, loader)
        trainer_state.logs.update({f"val_{k}": v for k, v in metrics.compute_metrics().items()})

class TestCallback(Callback):
    def on_epoch_start(self, trainer_state: TrainerState, **kwargs) -> None:
        """
        Evaluate the model on the validation set at the end of each epoch.
        """
        loader = trainer_state.data_module.test_dataloader()
        metrics = evaluate_model(trainer_state.model, loader)

        trainer_state.logs.update({f"test_{k}": v for k, v in metrics.compute_metrics().items()})
