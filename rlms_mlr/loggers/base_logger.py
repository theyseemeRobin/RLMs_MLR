from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union

import torch
from PIL import Image


LogMetric = Union[float, str, Image.Image]
class Logger(ABC):

    def log_metric(self, name: str, value: LogMetric, step: int):
        """
        Log a metric.
        Args:
            name: The name of the metric.
            value: The value of the metric.
            step: The step at which the metric is logged.

        Returns: None
        """
        if isinstance(value, float):
            self.log_float(name, value, step)
        elif isinstance(value, str):
            self.log_string(name, value, step)
        elif isinstance(value, Image.Image):
            self.log_image(name, value, step)
        else:
            raise ValueError(f"Unsupported metric type: {type(value)}. Supported types are: float, str, Image.Image")

    @abstractmethod
    def log_float(self, name: str, value: float, step: int):
        """
        Log a float metric.

        Args:
            name: The name of the metric.
            value: The value of the metric.
            step: The step at which the metric is logged.

        Returns: None
        """
        pass

    def log_string(self, name: str, value: str, step: int):
        """
        Log a string metric.

        Args:
            name: The name of the metric.
            value: The value of the metric.
            step: The step at which the metric is logged.

        Returns: None
        """
        pass

    def log_image(self, name: str, value: Image.Image, step: int):
        """
        Log an image metric.

        Args:
            name: The name of the metric.
            value: The value of the metric.
            step: The step at which the metric is logged.

        Returns: None
        """
        pass

    @abstractmethod
    def log_config(self, params: dict):
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

    def close(self):
        """
        Close the logger.
        Returns: None
        """
        pass