import logging
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter

from rlms_mlr.loggers.base_logger import Logger


class TensorBoardLogger(Logger):
    """
    Logger implementation for TensorBoard.
    """
    def __init__(self, log_dir: Path) -> None:
        """
        Initialize the TensorBoard logger.
        Args:
            log_dir: The directory where the logs will be saved.
        """
        self.writer = SummaryWriter(str(log_dir))
        logging.info(f"TensorBoard logs will be saved to {log_dir}")

    def log_metric(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.writer.add_text(f'param/{k}', str(v), 0)

    def save_model(self, model: torch.nn.Module, path: Path) -> None:
        torch.save(model.state_dict(), path)
        self.writer.add_text('checkpoint', f'Saved model at {path}', 0)

    def close(self) -> None:
        """
        Close the TensorBoard writer when the logger is deleted.
        """
        self.writer.close()