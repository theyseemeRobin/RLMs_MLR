from pathlib import Path
from typing import Optional, Dict, Any
import torch
import mlflow

from rlms_mlr.loggers.base_logger import Logger

class MLflowLogger(Logger):
    """
    Logger implementation for MLflow.
    """
    def __init__(
        self,
        experiment_name: str = "default",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_config(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, name: str, value: float, step: int) -> None:
        mlflow.log_metric(name, value, step)

    def save_model(self, model: torch.nn.Module, path: Path) -> None:
        """
        Save the model's state_dict to the specified path and log it to MLflow.

        Args:
            model: The model to save.
            path: The path where the model will be saved.

        Returns:

        """
        mlflow.pytorch.log_model(model, artifact_path=path)
        mlflow.log_artifact(str(path))