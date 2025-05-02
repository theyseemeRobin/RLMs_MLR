from abc import ABC, abstractmethod
from collections import UserDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset

from rlms_mlr.downloaders.base_downloader import Downloader
from rlms_mlr.utils.file_utils import get_cache_dir

@dataclass
class Batch(UserDict):
    """
    A data class to represent a batch of data. This class behaves like a dictionary,
    allowing for dictionary-like operations, while also storing inputs and targets separately.

    Attributes:
        inputs: A dictionary of input tensors, where keys are the names of the inputs and values are the tensors.
        targets: A dictionary of target tensors, where keys are the names of the targets and values are the tensors.
    """
    inputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize the underlying dictionary with inputs and targets
        self.data = {'inputs': self.inputs, 'targets': self.targets}

    def to_device(self, device: str):
        """
        Move the inputs and targets to the specified device.

        Args:
            device: The device to move the data to (e.g., 'cuda' or 'cpu').
        """
        for key in self.inputs:
            self.inputs[key] = self.inputs[key].to(device)
        for key in self.targets:
            self.targets[key] = self.targets[key].to(device)


class LocalDataset(Dataset, ABC):
    """
    Base class for local datasets. By providing a download source, the dataset can be downloaded and stored
    upon initialization. If the dataset is already present, it will be used directly.

    Args:
        data_dir: The directory where the dataset is stored (will be relative to the cache dir  when using cache).
        download_source: The source URL/token to download the dataset from.
        downloader: Implementation of the downloader protocol.
        use_cache: Whether to use the cache directory for storing the dataset. Cache directory is a system-dependent
                   directory as determined by :func: `rlms_mlr.utils.file_utils.get_cache_dir`.
    """
    def __init__(
            self,
            data_dir: Path,
            download_source: str = None,
            downloader: Downloader = None,
            use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        if use_cache:
            self.data_dir = get_cache_dir() / Path(data_dir)
        if not self.data_dir.exists():
            if download_source is None:
                raise ValueError(f"data_dir {self.data_dir} does not exist and no download source provided.")
            downloader(download_source, self.data_dir)

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the length of the dataset. This method should be implemented by subclasses.
        Returns:
            The length of the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, item: int) -> Batch:
        """
        Get a batch of data. This method should be implemented by subclasses.
        Args:
            item: The index of the item to get.
        Returns:
            A batch of data.
        """
        ...