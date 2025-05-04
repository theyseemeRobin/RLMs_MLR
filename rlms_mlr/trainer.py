from datetime import datetime
from pathlib import Path

import torch
from torch.amp import autocast

from rlms_mlr.callbacks.progress_bar_callback import RichProgressCallback
from rlms_mlr.augmentation.augmentation_pipeline import AugmentationPipeline
from rlms_mlr.callbacks.base_callback import Callback, TrainerState, CallbackList
from rlms_mlr.callbacks.eval_callback import ValidateCallback, TestCallback
from rlms_mlr.data.data_module import DataModule
from rlms_mlr.data.dataset import Batch
from rlms_mlr.models.base_model import Model


class Trainer:
    def __init__(
            self,
            model: Model,
            optimizer: torch.optim.Optimizer,
            data_module: DataModule,
            augmentation_pipeline: AugmentationPipeline = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            callback: Callback = None,
    ):
        """
        Trainer class for training and evaluating models.

        Args:
            model:
            optimizer:
            data_module:
            augmentation_pipeline:
            lr_scheduler:
            callback:
        """

        self.model = model
        # ensure the optimizer is created with the model parameters
        self.optimizer = optimizer.__class__(params=model.parameters(), **optimizer.defaults)
        self.augmentation_pipeline = augmentation_pipeline
        self.lr_scheduler = lr_scheduler
        self.data_module = data_module
        self.train_loader = data_module.train_dataloader()
        self.callback = callback or self.default_callback()

        # internal state
        self.trainer_context = TrainerState(
            model=self.model,
            optimizer=self.optimizer,
            data_module=self.data_module,
            total_training_batches=len(self.train_loader),
            stop_training=False,
        )

    def default_log_dir(self) -> Path:
        model = self.model.__class__.__name__
        data_name = (self.data_module.train_dataset.__class__.__name__ or
                    self.data_module.val_dataset.__class__.__name__ or
                    self.data_module.test_dataset.__class__.__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(model, data_name, timestamp)

    def default_callback(self) -> CallbackList:
        l = []
        l.append(RichProgressCallback())
        if self.data_module.val_dataset is not None and len(self.data_module.val_dataset):
            l.append(ValidateCallback())
        if self.data_module.test_dataset is not None and len(self.data_module.test_dataset):
            l.append(TestCallback())
        return CallbackList(l)


    def add_callback(self, callback: Callback):
        """
        Add a callback to the trainer. If the trainer already has a callback list, add the new callback to it.

        Args:
            callback:
        """
        if isinstance(self.callback, CallbackList):
            self.callback.add_callback(callback)
        else:
            self.callback = CallbackList([self.callback, callback])

    def _callback(self, method: str):
        """
        Call the callback method with the current trainer context. Process the returned CallbackReturn.
        Args:
            method: The callback method to call. This should be a method of the Callback class
                    (e.g. Callback.on_train_start).
        """
        if self.callback is None:
            return
        callback_return = getattr(self.callback, method)(self.trainer_context)
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


    def train(self, total_epochs: int, use_amp: bool = False, gradient_clip_norm: float = None):

        self.trainer_context.total_epochs = total_epochs
        self._callback("on_train_start")

        scaler = torch.amp.GradScaler(self.model.device, enabled=use_amp)
        for epoch in range(total_epochs):

            self.trainer_context.current_epoch = epoch
            self._callback("on_epoch_start")
            self.model.train()

            metrics = None
            for batch_idx, batch in enumerate(self.train_loader):
                self.trainer_context.current_batch = batch_idx
                self._callback("on_train_batch_start")
                new_metrics = self.train_step(batch, use_amp=use_amp, scaler=scaler,
                                            gradient_clip_norm=gradient_clip_norm)
                if metrics is None:
                    metrics = new_metrics
                else:
                    metrics.merge(new_metrics)

                self._callback("on_train_batch_end")

            self.trainer_context.logs.update({f"train_{k}": v for k, v in metrics.compute_metrics().items()})
            self._callback("on_epoch_end")
            if self.trainer_context.stop_training:
                break

        self._callback("on_train_end")

    def train_step(
            self,
            batch: Batch,
            use_amp: bool = False,
            scaler: torch.amp.GradScaler = None,
            gradient_clip_norm: float = None
    ):
        processed_batch = self._prepare_batch(batch)
        with autocast("cuda", enabled=use_amp):
            loss, metrics = self.model.compute_loss(processed_batch)
        scaler.scale(loss).backward()
        if gradient_clip_norm:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return metrics