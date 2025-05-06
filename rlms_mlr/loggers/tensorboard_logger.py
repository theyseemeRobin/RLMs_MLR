import logging
from pathlib import Path
from typing import Dict, Any
import pprint

import numpy as np
import torch
import yaml
from PIL import Image
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

    def log_float(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_string(self, name: str, value: str, step: int):
        self.writer.add_text(name, value, step)

    def log_image(self, name: str, value: Image.Image, step: int) -> None:
        rgb_value = value.convert("RGB")
        tensor_img = torch.from_numpy(np.array(rgb_value)).permute(2, 0, 1)
        self.writer.add_image(name, tensor_img, step)

    def log_config(self, cfg: Dict[str, Any]) -> None:
        yaml_str = yaml.dump(cfg)
        self.writer.add_text('config', yaml_str, 0)

    def save_model(self, model: torch.nn.Module, path: Path) -> None:
        model_path = self.writer.log_dir / path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        self.writer.add_text('checkpoint', f'Saved model at {model_path}', 0)

    def close(self) -> None:
        """
        Close the TensorBoard writer when the logger is deleted.
        """
        self.writer.close()