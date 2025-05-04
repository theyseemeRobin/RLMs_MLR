import logging
from typing import Dict, Union, Optional, Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from rlms_mlr.data.dataset import Batch


class DataModule:
    """
    DataModule for handling train, val, and test datasets.

    Args:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        test_dataset: The test dataset.
        batch_size: The batch size for the DataLoader.
        num_workers: Number of workers for DataLoader (default: 4).
        **dataloader_kwargs: Additional arguments used when constructing DataLoaders.
    """
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        **dataloader_kwargs,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.dataloader_kwargs = dataloader_kwargs


    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        train_split: float = 0.8,
        val_split: float = 0.2,
        test_split: float = 0,
        split_seed: int = 42,
        **dataloader_kwargs,
    ) -> "DataModule":
        """
        Create a DataModule from a single dataset, splitting it into train, val, and test sets, if specified.

        Args:
            dataset: The dataset to split.
            batch_size: The batch size for the DataLoader.
            train_split: The proportion of the dataset to use for training (default: 0.8).
            val_split: The proportion of the dataset to use for validation (default: 0.2).
            test_split: The proportion of the dataset to use for testing (default: 0).
            num_workers: Number of workers for DataLoader (default: 4).
            split_seed: Seed for random splitting (default: 42).
            **dataloader_kwargs: Additional arguments used when constructing DataLoaders.
        """
        splits = [train_split, val_split, test_split]
        if not abs(sum(splits) - 1.0) < 1e-6:
            raise ValueError("train/val/test splits must sum to 1.0")

        train_size = int(len(dataset) * train_split)
        val_size = int(len(dataset) * val_split)
        test_size = len(dataset) - train_size - val_size
        logging.debug(f"Train size: {train_size}, val: {val_size}, test: {test_size}")

        gen = torch.Generator().manual_seed(split_seed)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=gen,
        )
        return cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **dataloader_kwargs,
        )

    def _make_data_loader(self, dataset: Dataset, caller: str, **data_loader_kwargs) -> Optional[DataLoader]:
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset: The dataset to create a DataLoader for.
            caller: The name of the calling method (for logging purposes).
            **data_loader_kwargs: Additional arguments used when constructing DataLoader.

        Returns:
            DataLoader: The DataLoader for the dataset.
        """
        if dataset is None:
            logging.warning(f"Dataset is empty (called from {caller}). DataLoader will not be created.")
            return None
        merged_kwargs = {**self.dataloader_kwargs, **data_loader_kwargs}
        return DataLoader(
            dataset,
            **merged_kwargs,
        )

    def train_dataloader(self, **data_loader_kwargs) -> Optional[Iterator[Batch]]:
        """
        Create a DataLoader for the training dataset.

        Args:
            **data_loader_kwargs: Arguments used when constructing the DataLoader.
        """
        return self._make_data_loader(self.train_dataset, "train_dataloader", **data_loader_kwargs)

    def val_dataloader(self, **data_loader_kwargs) -> Optional[Iterator[Batch]]:
        """
        Create a DataLoader for the validation dataset.

        Args:
            **data_loader_kwargs: Arguments used when constructing the DataLoader.
        """
        return self._make_data_loader(self.val_dataset, "val_dataloader", **data_loader_kwargs)

    def test_dataloader(self, **data_loader_kwargs) -> Optional[Iterator[Batch]]:
        """
        Create a DataLoader for the test dataset.

        Args:
            **data_loader_kwargs: Arguments used when constructing the DataLoader.
        """
        return self._make_data_loader(self.test_dataset, "test_dataloader", **data_loader_kwargs)
