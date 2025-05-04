import logging
from pathlib import Path
from typing import Dict, Any
import pprint

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from rlms_mlr.loggers.base_logger import Logger


class StdOutLogger(Logger):
    """
    Logger implementation for TensorBoard.
    """
    def __init__(self):
        self.metrics = {}
        self.last_step = 0

    def log_float(self, name: str, value: float, step: int) -> None:
        if step != self.last_step:
            print(f"Step: {self.last_step}\n")
            self.last_step = step
        print(f"{name}: {value:.4f}", end=" | ")

    def log_string(self, name: str, value: str, step: int) -> None:
        print(f"{name}: {value} | {step}")

    def log_config(self, cfg: Dict[str, Any]) -> None:
        yaml_str = yaml.dump(cfg)
        print("Configuration:")
        print(yaml_str)

    def save_model(self, model: torch.nn.Module, path: Path) -> None:
        logging.warning(f"Saving model to {path} is not supported in StdOutLogger.")