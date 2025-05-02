from typing import Dict

import torch

from rlms_mlr.callbacks.base_callback import Logs
from rlms_mlr.models.base_model import Model


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

    def compute_loss(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred = self.forward(image)
        loss = torch.nn.functional.cross_entropy(pred, label)
        return loss

    def evaluate(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
        pred = self.forward(image)
        loss = torch.nn.functional.cross_entropy(pred, label)
        acc = (pred.argmax(dim=1) == label).float().mean()
        return {'eval_loss' : loss.item(), 'accuracy' : acc.item()}