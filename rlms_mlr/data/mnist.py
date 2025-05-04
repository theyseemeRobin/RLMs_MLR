from torchvision.datasets import MNIST
from rlms_mlr.data.image_folder import ImageBatch


class MnistDataset(MNIST):

    def __getitem__(self, item: int) -> ImageBatch:
        img, label = super().__getitem__(item)
        return ImageBatch(images=img, labels=label)