from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Dict
from torch import optim

from rlms_mlr.data import DataModule
from rlms_mlr.models.base_model import Model


@dataclass
class TrainerState:
    """
    Context object passed to callbacks during training.

    Attributes:
        model: The model being trained.
        optimizer: The optimizer used for training.
        total_epochs: Total number of epochs to train.
        total_training_batches: Total number of training batches.
        current_epoch: Current epoch number.
        current_batch: Current batch number.
        stop_training: Boolean indicating whether to stop training or not.
        logs: Logs object containing training and evaluation losses.
    """
    model: Model
    optimizer: optim.Optimizer
    data_module: DataModule
    total_epochs: int
    total_training_batches: int
    current_epoch: int
    current_batch: int
    stop_training: bool = False
    logs: Dict[str, float] = None


@dataclass
class CallbackReturn:
    """
    Return object from callbacks.

    Attributes:
        stop_training: bool: Boolean indicating whether to stop training or not.
    """
    stop_training: bool = False

    def update(self, other: "CallbackReturn"):
        """
        Update the current CallbackReturn object with another one.
        Args:
            other: Another CallbackReturn object.
        Returns: None
        """
        self.stop_training = other.stop_training


class Callback(ABC):
    """
    Base class for all callbacks. Callbacks are used to hook into the training process and perform actions at various
    stages of training. Each callback can define its own priority, which determines the order in which they are called
    when multiple callbacks are used in a CallbackList.
    """
    priority: int = 0

    def on_train_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the start of training.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_epoch_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the start of each epoch.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_train_batch_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the start of each training batch.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_validation_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the start of validation.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_validation_batch_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the start of each validation batch.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...

    def on_train_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the end of training.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_epoch_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the end of each epoch.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_train_batch_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the end of each training batch.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_validation_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the end of validation.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...
    def on_validation_batch_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        """
        Hook into the training process. Called at the end of each validation batch.
        Args:
            trainer_state: TrainerState object containing the current state of the training process.
            **kwargs: Additional keyword arguments that can be passed to the callback, if needed.
        Returns: None
        """
        ...


class CallbackList(Callback):

    def __init__(self, callbacks: List[Callback] = None):
        """
        Initialize the CallbackList with a list of callbacks. The callbacks are called in sequential order based on
        their priority.

        Args:
            callbacks: A list of Callback instances.
        """
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, list) or not all(isinstance(cb, Callback) for cb in callbacks):
            raise ValueError("callbacks must be a list of Callback instances")
        self.callbacks = callbacks
        self.sort_callbacks()

    def sort_callbacks(self):
        """
        Sort the callbacks in the list based on their priority.
        """
        self.callbacks = sorted(self.callbacks, key=lambda c: c.priority)

    def add_callback(self, callback: Callback):
        """
        Add a callback to the list of callbacks.

        Args:
            callback: A Callback instance to add.
        """
        if not isinstance(callback, Callback):
            raise ValueError("callback must be a Callback instance")
        self.callbacks.append(callback)
        self.sort_callbacks()

    def remove_callback(self, callback_type: type):
        """
        Remove a callback from the list of callbacks.

        Args:
            callback_type: Type of the callback to remove.
        """
        callbacks = []
        for cb in self.callbacks:
            if not isinstance(cb, callback_type):
                callbacks.append(cb)
        self.callbacks = callbacks
        self.sort_callbacks()

    def callback_all(self, func_name: str, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        if len(self.callbacks) == 0:
            return None
        callback_return = CallbackReturn()
        for callback in self.callbacks:
            func = getattr(callback, func_name, None)
            if callable(func):
                ret = func(trainer_state, **kwargs)
                if ret is not None:
                    callback_return.update(ret)
        return callback_return

    def on_train_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_train_start", trainer_state=trainer_state, **kwargs)

    def on_epoch_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_epoch_start", trainer_state=trainer_state, **kwargs)

    def on_train_batch_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_train_batch_start", trainer_state=trainer_state, **kwargs)

    def on_validation_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_validation_start", trainer_state=trainer_state, **kwargs)

    def on_validation_batch_start(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_validation_batch_start", trainer_state=trainer_state, **kwargs)

    def on_train_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_train_end", trainer_state=trainer_state, **kwargs)

    def on_epoch_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_epoch_end", trainer_state=trainer_state, **kwargs)

    def on_train_batch_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_train_batch_end", trainer_state=trainer_state, **kwargs)

    def on_validation_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_validation_end", trainer_state=trainer_state, **kwargs)

    def on_validation_batch_end(self, trainer_state: TrainerState, **kwargs) -> Optional[CallbackReturn]:
        self.callback_all(func_name="on_validation_batch_end", trainer_state=trainer_state, **kwargs)
