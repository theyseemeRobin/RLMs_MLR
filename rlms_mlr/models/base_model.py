from abc import ABC, abstractmethod
from typing import Dict
import torch.nn

from rlms_mlr.data.dataset import Batch


class Model(torch.nn.Module, ABC):
    """
    Base class for all models. All models should inherit from this class.
    """

    # TODO: I don't like using dictionaries for inputs and targets. This should be changed to a more structured
    #       approach, such as using dataclasses or named tuples.
    @abstractmethod
    def compute_loss(self, batch: Batch) -> torch.Tensor:
        """
        Perform a forward pass on a batch and compute the loss for the model. This method should be implemented by
        subclasses.

        Args:
            batch: A Batch object containing input and target tensors.

        Returns: Loss value as a tensor.
        """
        ...

    @abstractmethod
    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def evaluate(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Evaluate the model on a batch of data. This method should be implemented by subclasses.

        Args:
            batch: A Batch object containing input and target tensors.

        Returns: A dictionary of evaluation metrics, such as evaluation loss, accuracy, etc.
        """
        ...