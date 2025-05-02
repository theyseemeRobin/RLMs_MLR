from abc import ABC, abstractmethod
from pathlib import Path

import torch


class Logger(ABC):

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int):
        """
        Log a metric.
        Args:
            name: The name of the metric.
            value: The value of the metric.
            step: The step at which the metric is logged.

        Returns: None

        """
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict):
        """
        Log hyperparameters.

        Args:
            params: A dictionary of hyperparameters.

        Returns: None

        """
        pass

    @abstractmethod
    def save_model(self, model: torch.nn.Module, path: Path) -> None:
        """
        Save the model.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        Returns: None
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the logger.
        Returns: None
        """
        pass