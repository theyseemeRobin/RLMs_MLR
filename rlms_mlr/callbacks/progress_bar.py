from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from rlms_mlr.callbacks.base_callback import Callback, TrainerState



class TqdmProgressCallback(Callback):
    """
    Displays a simple tqdm progress bar for training.

    Args:
        **bar_kwargs: Additional arguments to pass to tqdm.
    """
    priority = 0

    def __init__(self, **bar_kwargs):
        self.training_bar = None
        self.bar_kwargs = bar_kwargs

    def on_train_start(self, trainer_state: TrainerState, **kwargs) -> None:
        total_batches = trainer_state.total_epochs * trainer_state.total_training_batches
        self.training_bar = tqdm(total=total_batches, desc="Training (tqdm)", leave=True, **self.bar_kwargs)

    def on_train_batch_end(self, trainer_state: TrainerState, **kwargs) -> None:
        self.training_bar.update(1)

    def on_train_end(self, trainer_state: TrainerState, **kwargs) -> None:
        self.training_bar.close()

class RichProgressCallback(Callback):
    """
    Displays an enhanced progress bar using rich.progress.
    Args:
        **bar_kwargs: Additional arguments to pass to rich.progress.
    """
    priority = 0
    def __init__(self, **bar_kwargs):
        self.progress = None
        self.task_id = None
        self.bar_kwargs = bar_kwargs

    def on_train_start(self, trainer_state: TrainerState, **kwargs) -> None:
        total_batches = trainer_state.total_epochs * trainer_state.total_training_batches
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            expand=True,
            **self.bar_kwargs
        )
        self.progress = progress.__enter__()
        self.task_id = self.progress.add_task("Training (rich)", total=total_batches)

    def on_train_batch_end(self, trainer_state: TrainerState, **kwargs) -> None:
        self.progress.update(self.task_id, advance=1, update=True)
        self.progress.refresh()

    def on_train_end(self, trainer_state: TrainerState, **kwargs) -> None:
        self.progress.__exit__(None, None, None)
