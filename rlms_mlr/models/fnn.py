import io
from typing import TypedDict, Tuple
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
)

from rlms_mlr.data.image_folder import ImageBatch
from rlms_mlr.models.base_model import Model, Metrics





class ClassificationMetricsDict(TypedDict):
    accuracy: float
    loss: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: Image.Image


class ClassificationMetrics(Metrics):

    def compute_metrics(self) -> ClassificationMetricsDict:
        # Move tensors to CPU and convert to numpy
        y_pred = self.predictions.detach().cpu().numpy()
        y_true = self.ground_truths.detach().cpu().numpy()

        # Accuracy
        acc = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1 (weighted for multi-class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        display.plot(ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)
        buf.close()


        return ClassificationMetricsDict(
            accuracy=acc,
            loss=float(self.loss.item()),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            confusion_matrix=img,
        )


class FNN(Model):
    def __init__(
            self,
            fnn: torch.nn.Module,
            device: str,
            state_dict: dict = None,
    ):
        super(FNN, self).__init__()
        self.fnn = fnn
        # TODO: make device accessible in the base class
        self.device = device
        self.fnn.to(self.device)
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.fnn(x)
        return x

    def compute_loss(self, image_batch: ImageBatch) -> Tuple[torch.Tensor, ClassificationMetrics]:
        pred = self.forward(image_batch.images)
        loss = torch.nn.functional.cross_entropy(pred, image_batch.labels)
        return loss, ClassificationMetrics(predictions=pred.argmax(dim=1), ground_truths=image_batch.labels, loss=loss)

    def evaluate(self, image_batch: ImageBatch) -> ClassificationMetrics:
        pred = self.forward(image_batch.images)
        loss = torch.nn.functional.cross_entropy(pred, image_batch.labels)
        pred_labels = pred.argmax(dim=1)
        ground_truth_labels = image_batch.labels
        return ClassificationMetrics(predictions=pred_labels, ground_truths=ground_truth_labels, loss=loss)