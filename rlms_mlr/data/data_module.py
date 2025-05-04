import logging
from typing import Optional, Iterator

import torch
from torch.utils.data import DataLoader

from rlms_mlr.data.dataset import Batch, BaseDataset


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
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
        test_dataset: BaseDataset,
        **dataloader_kwargs,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.dataloader_kwargs = dataloader_kwargs


    @classmethod
    def from_dataset(
        cls,
        dataset: BaseDataset,
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

        logging.debug(f"Train split: {train_split}, val split: {val_split}, test split: {test_split}")

        gen = torch.Generator().manual_seed(split_seed)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_split, val_split, test_split],
            generator=gen,
        )
        return cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **dataloader_kwargs,
        )

    def _make_data_loader(self, dataset: BaseDataset, caller: str, **data_loader_kwargs) -> DataLoader:
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
            raise ValueError(f"Dataset is empty (called from {caller}) - cannot create DataLoader.")
        merged_kwargs = {**self.dataloader_kwargs, **data_loader_kwargs}
        return DataLoader(
            dataset,
            **merged_kwargs,
        )

    def train_dataloader(self, **data_loader_kwargs) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        Args:
            **data_loader_kwargs: Arguments used when constructing the DataLoader.
        """
        return self._make_data_loader(self.train_dataset, "train_dataloader", **data_loader_kwargs)

    def val_dataloader(self, **data_loader_kwargs) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        Args:
            **data_loader_kwargs: Arguments used when constructing the DataLoader.
        """
        return self._make_data_loader(self.val_dataset, "val_dataloader", **data_loader_kwargs)

    def test_dataloader(self, **data_loader_kwargs) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Args:
            **data_loader_kwargs: Arguments used when constructing the DataLoader.
        """
        return self._make_data_loader(self.test_dataset, "test_dataloader", **data_loader_kwargs)
