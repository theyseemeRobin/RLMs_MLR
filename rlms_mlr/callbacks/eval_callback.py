import torch

from rlms_mlr.callbacks.base_callback import Callback, TrainerState, CallbackReturn, Logs


class EvalCallback(Callback):

    def __init__(self, eval_loader):
        self.eval_loader = eval_loader

    def on_epoch_end(self, trainer_state: TrainerState, **kwargs) -> None:
        """
        Evaluate the model on the validation set at the end of each epoch.
        """
        trainer_state.model.eval()
        eval_loss = 0

        mean_metrics = {}
        trainer_state.model.eval()
        for batch in self.eval_loader:
            batch.to_device(trainer_state.model.device)
            with torch.no_grad():
                metrics = trainer_state.model.evaluate(**batch.inputs, **batch.targets)
            for key, value in metrics.items():
                if key not in mean_metrics:
                    mean_metrics[key] = 0
                # iterative mean
                mean_metrics[key] += value / len(self.eval_loader)

        eval_loss /= len(self.eval_loader)
        trainer_state.logs = Logs(**mean_metrics)