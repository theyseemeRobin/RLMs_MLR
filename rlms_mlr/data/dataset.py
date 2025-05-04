from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Union
import torch
from torch.utils.data import Dataset

from rlms_mlr.downloaders.base_downloader import Downloader
from rlms_mlr.utils.file_utils import get_cache_dir


class Batch(dict):
    """
    Any dict whose values are torch.Tensors and keys are strings. Values can also be retrieved as attributes. Since
    this is a dict, torch dataloader will batchify it automatically.

    Examples:
        >>> batch = Batch(images=torch.randn(2, 3, 224, 224), labels=torch.tensor([0, 1]))
        >>> batch['images'].shape # torch.Size([2, 3, 224, 224])
        >>> batch.images.shape    # torch.Size([2, 3, 224, 224])
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key: str) -> torch.Tensor:
        return super().__getitem__(key)

    def __getattr__(self, key: str) -> torch.Tensor:
        if super().__getattribute__(key) is not None:
            return super().__getattribute__(key)
        return self.__getitem__(key)

    def keys(self) -> Mapping[str, torch.Tensor].keys:
        return super().keys()

    def items(self) -> Mapping[str, torch.Tensor].items:
        return super().items()

    def to(self, device: Union[str, torch.device], **kwargs) -> 'Batch':
        return self.__class__(**{key: value.to(device, **kwargs) for key, value in self.items()})


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
        # while this is technically not a batch but a sample, torch loaders will batchify it, since Batch is a dict
        # and torch loaders will batchify dicts
        ...