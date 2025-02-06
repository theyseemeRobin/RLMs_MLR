from dataclasses import dataclass, field
import os.path
from typing import Optional, Union

import torch
import omegaconf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import MISSING


@dataclass
class Train:
    datapath: str = MISSING
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_savedir: Optional[str] = 'weights'

    def initialize(self):
        self.datapath = to_absolute_path(self.datapath)
        # if not os.path.exists(self.datapath):
        #     raise FileNotFoundError(f"Data path {self.datapath} not found.")
        if self.weight_savedir.startswith('$'):
            self.weight_savedir = to_absolute_path(self.weight_savedir)
        os.makedirs(self.weight_savedir, exist_ok=True)


@dataclass
class Evaluate:
    datapath: str = MISSING
    weights: Optional[str] = 'weights/final.pth'

    def initialize(self):
        self.datapath = to_absolute_path(self.datapath)
        # if not os.path.exists(self.datapath):
        #     raise FileNotFoundError(f"Data path {self.datapath} not found.")
        if self.weights.startswith('$'):
            self.weights = to_absolute_path(self.weights)


@dataclass
class Visualize:
    datapath: str = MISSING
    weights: Optional[str] = 'weights/final.pth'

    def initialize(self):
        self.datapath = to_absolute_path(self.datapath)
        # if not os.path.exists(self.datapath):
        #     raise FileNotFoundError(f"Data path {self.datapath} not found.")
        if self.weights.startswith('$'):
            self.weights = to_absolute_path(self.weights)


@dataclass
class System:
    device: str = 'cpu'
    log_path: str = 'logs'

    def initialize(self):

        if self.log_path.startswith('$'):
            self.log_path = to_absolute_path(self.log_path)
        os.makedirs(self.log_path, exist_ok=True)

        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            print('GPU is not available, using CPU instead.')
        else:
            self.device = 'cpu'


@dataclass
class Config:
    system: System = field(default_factory=System)
    train: Optional[Train] = None
    evaluate: Optional[Evaluate] = None
    visualize: Optional[Visualize] = None

    def initialize(self):
        self.system.initialize()
        if self.train is not None:
            self.train.initialize()
        if self.evaluate is not None:
            self.evaluate.initialize()
        if self.visualize is not None:
            self.visualize.initialize()

        print(
            f"Running experiment with Config:\n"
            f"{'Started from':<28} {get_original_cwd():20}\n"
            f"{'Current working directory:':<28} {os.getcwd()}\n\n"
            f"Configuration: \n{self}"
        )
        self.visualize.weights = 'banana for test'

    def __str__(self):
        return omegaconf.OmegaConf.to_yaml(self)


cs = ConfigStore.instance()
cs.store(name="base_evaluate", node=Evaluate, group='evaluate')
cs.store(name="base_train", node=Train, group='train')
cs.store(name="base_visualize", node=Visualize, group='visualize')
cs.store(name="base_system", node=System, group='system')
cs.store(name="base_config", node=Config)