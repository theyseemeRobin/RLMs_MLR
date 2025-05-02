from abc import ABC, abstractmethod
from typing import Dict
import torch.nn


class Model(torch.nn.Module, ABC):
    """
    Base class for all models. All models should inherit from this class.
    """

    # TODO: I don't like using dictionaries for inputs and targets. This should be changed to a more structured
    #       approach, such as using dataclasses or named tuples.
    @abstractmethod
    def compute_loss(self, inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a forward pass on a batch and compute the loss for the model. This method should be implemented by
        subclasses.

        Args:
            inputs: A dictionary of input tensors, where keys are the names of the inputs and values are the tensors.
            targets: A dictionary of target tensors, where keys are the names of the targets and values are the tensors.

        Returns: Loss value as a tensor.
        """
        ...

    @abstractmethod
    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def evaluate(self, inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate the model on a batch of data. This method should be implemented by subclasses.

        Args:
            inputs: A dictionary of input tensors, where keys are the names of the inputs and values are the tensors.
            targets: A dictionary of target tensors, where keys are the names of the targets and values are the tensors.

        Returns: A dictionary of evaluation metrics, such as evaluation loss, accuracy, etc.

        """
        ...