from pathlib import Path
from typing import Optional, Dict, Any
import torch
import wandb

from rlms_mlr.loggers.base_logger import Logger

# TODO: update for actual config object
class WandbLogger(Logger):
    """
    Logger implementation for Weights & Biases (W&B).
    """
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.run = wandb.init(project=project, entity=entity, config=config)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        wandb.config.update(params)

    def log_metric(self, name: str, value: float, step: int) -> None:
        wandb.log({name: value}, step=step)

    def save_model(self, model: torch.nn.Module, path: Path) -> None:
        """
        Save the model's state_dict to the specified path and upload it to W&B.
        Args:
            model: The model to save.
            path: The path where the model will be saved.
        Returns: None
        """
        torch.save(model.state_dict(), path)
        wandb.save(path)

    def close(self) -> None:
        """Finish the W&B run when the logger is deleted."""
        wandb.finish()
