from abc import abstractmethod, ABC
import torch
import torch.nn
from typing import Dict

from rlms_mlr.data.dataset import Batch





class Metrics(ABC):
    def __init__(
            self,
            predictions: torch.Tensor = torch.empty(0),
            ground_truths: torch.Tensor = torch.empty(0),
            loss: float = 0,
    ):
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.loss = loss
        self._n_updates = 0

    def extend(self, predictions: torch.Tensor, ground_truths: torch.Tensor, loss: float):
        """
        Extend the metrics with new predictions and ground truths. Loss is assumed to be the loss of a single batch,
        with the final loss being the mean of all batches, disregarding the batch sizes.

        Args:
            predictions: The predictions of the model.
            ground_truths: The ground truths of the model.
            loss: The loss of the model.
        """
        self._n_updates += 1
        self.predictions = torch.cat((self.predictions, predictions), dim=0)
        self.ground_truths = torch.cat((self.ground_truths, ground_truths), dim=0)
        self.loss += (loss - self.loss) / self._n_updates

    def merge(self, other: 'Metrics'):
        """
        Merge the metrics with another metrics object.

        Args:
            other: The other metrics object to merge with.
        """
        self.extend(other.predictions, other.ground_truths, other.loss)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute the metrics from the predictions and ground truths.

        Returns: A dictionary of metrics.
        """
        ...


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
    def evaluate(self, batch: Batch) -> Metrics:
        """
        Evaluate the model on a batch of data. This method should be implemented by subclasses.

        Args:
            batch: A Batch object containing input and target tensors.

        Returns: A dictionary of evaluation metrics, such as evaluation loss, accuracy, etc.
        """
        ...