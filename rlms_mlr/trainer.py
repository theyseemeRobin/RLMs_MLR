from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.amp import autocast
from typing import Any, Optional, Callable, Union, Sequence, Iterable, List

from rlms_mlr.callbacks.logging import LoggerCallback
from rlms_mlr.callbacks.progress_bar import RichProgressCallback
from rlms_mlr.augmentation.augmentation_pipeline import AugmentationPipeline
from rlms_mlr.callbacks.base_callback import Callback, TrainerState, Logs, CallbackList
from rlms_mlr.callbacks.eval_callback import EvalCallback
from rlms_mlr.data.data_module import DataModule
from rlms_mlr.data.dataset import Batch
from rlms_mlr.loggers.base_logger import Logger
from rlms_mlr.loggers.tensorboard_logger import TensorBoardLogger
from rlms_mlr.models.base_model import Model


@dataclass
class ModelConfig:
    _target_: str
    _args_: dict = None
    _kwargs_: dict = None

@dataclass
class TrainerConfig:
    """
    Configuration class for the Trainer. This class is used to store the configuration parameters for the Trainer.

    Attributes:
    """

    # objects
    model: Model
    data_module: DataModule
    device: str = "cuda"
    augmentation_pipeline: AugmentationPipeline = None
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    optimizer_cls: torch.optim.Optimizer = torch.optim.Adam

    # training parameters
    total_epochs: int = 10
    use_amp: bool = False
    gradient_clip_norm: float = None
    optimizer_kwargs: dict = None
    loader_kwargs: dict = None

    # logging
    logger: Logger = None
    log_dir: Path = None

    # callbacks
    callbacks: List[Callback] = None
    validate: bool = True
    test: bool = False
    progress_bar_callback = RichProgressCallback()

    def __post_init__(self):
        if self.test and self.validate:
            # TODO: change the Logs propagation of callbacks to fix this issue
            raise RuntimeError("Both test and validate cannot be True at the same time.")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return asdict(self).keys()

    def generate_path(self) -> Path:
        """
        Generate a unique path based on the model name, dataset name, and current timestamp.

        Returns: A Path object representing the generated path.
        """
        model = self.model.__class__.__name__
        data_name = (self.data_module.train_dataset.__class__.__name__ or
                    self.data_module.val_dataset.__class__.__name__ or
                    self.data_module.test_dataset.__class__.__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(model, data_name, timestamp)

    def get_trainer(self) -> "Trainer":
        """
        Create a Trainer instance using the configuration parameters.
        Returns:
            Trainer: An instance of the Trainer class.
        """
        self.data_module.dataloader_kwargs = self.loader_kwargs or {}

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}

        # Create the optimizer
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

        callbacks = self.callbacks or []
        if self.validate:
            callbacks.append(EvalCallback(self.data_module.val_dataloader()))
        if self.test:
            callbacks.append(EvalCallback(self.data_module.test_dataloader()))
        self.log_dir = self.log_dir or self.generate_path()
        logger = self.logger or TensorBoardLogger(self.log_dir)
        log_callback = LoggerCallback(logger=logger, params=dict(self), log_dir=self.log_dir)
        callbacks.append(log_callback)
        if self.progress_bar_callback:
            callbacks.append(self.progress_bar_callback)

        return Trainer(
            model=self.model,
            optimizer=optimizer,
            data_module=self.data_module,
            total_epochs=self.total_epochs,
            augmentation_pipeline=self.augmentation_pipeline,
            lr_scheduler=self.lr_scheduler,
            callbacks=CallbackList(callbacks),
            use_amp=self.use_amp,
            gradient_clip_norm=self.gradient_clip_norm,
        )


class Trainer:
    def __init__(
            self,
            model: Model,
            optimizer: torch.optim.Optimizer,
            data_module: DataModule,
            total_epochs: int,
            augmentation_pipeline: AugmentationPipeline = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            callbacks: Callback = None,
            use_amp: bool = False,
            gradient_clip_norm: float = None,
    ):

        # Training objects
        self.model = model
        self.optimizer = optimizer
        self.augmentation_pipeline = augmentation_pipeline
        self.lr_scheduler = lr_scheduler

        # training configuration
        self.total_epochs = total_epochs
        self.use_amp = use_amp
        self.gradient_clip_norm = gradient_clip_norm
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # data
        self.data_module = data_module
        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()
        self.test_loader = data_module.test_dataloader()

        self.callbacks = callbacks

        # internal state
        self.trainer_context = TrainerState(
            model=self.model,
            optimizer=self.optimizer,
            data_module=self.data_module,
            total_epochs=self.total_epochs,
            total_training_batches=len(self.train_loader),
            current_epoch=0,
            current_batch=0,
            stop_training=False,
        )
    def _callback(self, method: str):
        """
        Call the callback method with the current trainer context. Process the returned CallbackReturn.
        Args:
            method: The callback method to call. This should be a method of the Callback class
                    (e.g. Callback.on_train_start).
        """
        if self.callbacks is None:
            return
        callback_return = getattr(self.callbacks, method)(self.trainer_context)
        if callback_return is None:
            return

        if callback_return.stop_training:
            self.trainer_context.stop_training = True

    def _prepare_batch(
            self,
            batch: Batch,
    ) -> Batch:
        batch = batch.to(device=self.model.device, non_blocking=True)
        if self.augmentation_pipeline:
            batch = self.augmentation_pipeline(batch)
        return batch


    def train(self):
        self._callback("on_train_start")

        for epoch in range(self.total_epochs):
            self.trainer_context.current_epoch = epoch
            self._callback("on_epoch_start")

            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                self.trainer_context.current_batch = batch_idx
                self._callback("on_train_batch_start")

                batch_prepared = self._prepare_batch(batch)
                with autocast("cuda", enabled=self.use_amp):
                    loss = self.model.compute_loss(batch_prepared)
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.lr_scheduler:
                    self.lr_scheduler.step()

                self.trainer_context.logs = Logs(loss=loss)
                self._callback("on_train_batch_end")

            self._callback("on_epoch_end")

            if self.trainer_context.stop_training:
                break
        self._callback("on_train_end")