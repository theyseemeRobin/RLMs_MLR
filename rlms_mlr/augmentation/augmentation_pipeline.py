from abc import ABC, abstractmethod

from rlms_mlr.data.dataset import Batch


class AugmentationPipeline(ABC):
    @abstractmethod
    def __call__(self, batch: Batch) -> Batch:
        """
        Apply the augmentation pipeline to a batch of data.

        Args:
            batch: A batch of data to augment.

        Returns:
            The augmented batch.
        """
        pass

class AugmentationPipelineSequence(AugmentationPipeline):
    def __init__(self, augmenters: list[AugmentationPipeline]):
        """
        Initialize the augmentation pipeline with a sequence of augmenters.

        Args:
            augmenters: A list of augmentation functions to apply in sequence.
        """
        self.augmenters = augmenters

    def __call__(self, batch: Batch) -> Batch:
        """
        Apply the augmentation pipeline to a batch of data.

        Args:
            batch: A batch of data to augment.

        Returns:
            The augmented batch.
        """
        for augmenter in self.augmenters:
            batch = augmenter(batch)
        return batch