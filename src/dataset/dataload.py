import os
import abc
import glob
import math
import logging
import numpy as np

from skimage.io import imread
import PIL
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95

DATASETS_DICT = {"openimages": "OpenImages", "cityscapes": "CityScapes",
                 "jetimages": "JetImages", "evaluation": "Evaluation"}
DATASETS = list(DATASETS_DICT.keys())


class BaseDataset(Dataset):
    """Base Class for datasets.
    Parameters
    ----------
    root : string
        Root directory of dataset.
    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], mode='train', logger=logging.getLogger(__name__),
                 **kwargs):
        self.root = root

        try:
            self.train_data = os.path.join(root, self.files["train"])
            self.test_data = os.path.join(root, self.files["test"])
            self.val_data = os.path.join(root, self.files["val"])
        except AttributeError:
            pass

        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            raise ValueError('Files not found in specified directory: {}'.format(root))

    def __len__(self):
        return len(self.imgs)

    def __ndim__(self):
        return tuple(self.imgs.size())

    def __getitem__(self, idx):
        """Get the image of `idx`.
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass


class Evaluation(BaseDataset):
    """
    Parameters
    ----------
    root : string
        Root directory of dataset.
    """
    def __init__(self, root=os.path.join(DIR, "data"), normalize=False, **kwargs):
        super(Evaluation, self).__init__()

        self.imgs = glob.glob(os.path.join(root, "*.jpg"))
        self.imgs += glob.glob(os.path.join(root, "*.png"))

        self.normalize = normalize

    def _transform(self):
        transforms_list = [transforms.ToTensor()]

        if self.normalize:
            transforms_list += [
                # transforms.Resize(256),
                # transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        return transforms.Compose(transforms_list)

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unknown dataset: {}".format(dataset))


def get_img_size(dataset):
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def exception_collate_fn(batch):
    exception_batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(exception_batch)


def get_dataloader(dataset, mode="train", root=None, shuffle=True, pin_memory=True, batch_size=8,
                   logger=logging.getLogger(__name__), normalize=False, **kwargs):
    """
    A generic data loader
        Parameters
        ----------
        dataset : {"openimages", "jetimages", "evaluation"}
            Name of the dataset to load
        root : str
            Path to the dataset root. If `None` uses the default one.
        kwargs :
            Additional arguments to `DataLoader`. Default values are modified.
    """

    pin_memory = pin_memory and torch.cuda.is_available()   # Only set pin_memory True if GPU is available

    if root is None:
        dataset = Dataset(logger=logger, mode=mode, normalize=normalize, **kwargs)
    else:
        dataset = Dataset(root=root, logger=logger, mode=mode, normalize=normalize, **kwargs)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=NUM_DATASET_WORKERS,
                      collate_fn=exception_collate_fn,
                      pin_memory=pin_memory
                      )

if __name__ == "__main__":
    pass