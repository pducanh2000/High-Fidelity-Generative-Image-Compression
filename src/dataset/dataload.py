import os
import math
import json
import logging
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95

DATASETS_DICT = {"openimages": "OpenImages", "cityscapes": "CityScapes",
                 "jetimages": "JetImages", "evaluation": "Evaluation",
                 "kodak": "KodakDataset", "vimeo": "VimeoDataset"}
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

    def __init__(self, json_path, logger=logging.getLogger(__name__),
                 **kwargs):
        self.json_path = json_path
        self.imgs = None

        self.logger = logger

        if not os.path.exists(self.json_path):
            raise ValueError('Files not found in specified directory: {}'.format(self.json_path))

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

    def __init__(self, json_path, normalize=False, **kwargs):
        super(Evaluation, self).__init__(json_path=json_path, **kwargs)

        with open(json_path, "r") as f:
            self.eval_imgs = json.load(f)
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

    def __getitem__(self, item):
        img_path = self.eval_imgs[item]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        file_size = os.path.getsize(img_path)

        try:
            img = Image.open(img_path)
            img = img.convert('RGB')
            W, H = img.size  # slightly confusing
            bpp = file_size * 8. / (H * W)

            transformation = self._transform()
            transformed = transformation(img)
        except:
            print('Error reading input images!')
            return None

        return transformed, bpp, filename


class KodakDataset(BaseDataset):
    def __init__(self,
                 json_path=os.path.abspath(os.path.join(DIR, "../../data/kodak_dataset/data.json")),
                 crop_size=256,
                 normalize=False,
                 **kwargs
                 ):
        super(KodakDataset, self).__init__(json_path=json_path, **kwargs)

        with open(json_path, "r") as f:
            self.imgs = json.load(f)

        self.crop_size = crop_size
        self.image_dims = (3, self.crop_size, self.crop_size)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX
        self.normalize = normalize

    def _transform(self, scale, H, W):
        transforms_list = [transforms.ToTensor()]
        if self.normalize:
            transforms_list = [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
                transforms.RandomCrop(self.crop_size),
                transforms.ToTensor()
            ]
        return transforms.Compose(transforms_list)

    def __getitem__(self, item):
        image_path = self.imgs[item]
        file_size = os.path.getsize(image_path)

        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            W, H = img.size  # slightly confusing
            bpp = file_size * 8. / (H * W)

            shortest_size_length = min(H, W)
            minimum_scale_factor = 1.0 * self.crop_size / float(shortest_size_length)
            scale_low = max(minimum_scale_factor, self.scale_min)
            scale_high = max(scale_low, self.scale_max)
            scale = np.random.uniform(scale_low, scale_high)

            dynamic_transform = self._transform(scale, H, W)
            transformed = dynamic_transform(img)

        except:
            return None

        return transformed, bpp


class VimeoDataset(BaseDataset):
    def __init__(self,
                 json_path=os.path.abspath(os.path.join(DIR, "../../data/vimeo_interp_test/data.json")),
                 crop_size=256,
                 normalize=False,
                 train_mode=True,
                 **kwargs
                 ):
        super(VimeoDataset, self).__init__(json_path=json_path, **kwargs)

        with open(json_path, "r") as f:
            data_dict = json.load(f)
            self.imgs = data_dict["train"] if train_mode else data_dict["val"]

        self.crop_size = crop_size
        self.image_dims = (3, self.crop_size, self.crop_size)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX
        self.normalize = normalize

    def _transform(self, scale, H, W):
        transforms_list = [transforms.ToTensor()]
        if self.normalize:
            transforms_list = [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
                transforms.RandomCrop(self.crop_size),
                transforms.ToTensor()
            ]
        return transforms.Compose(transforms_list)

    def __getitem__(self, item):
        image_path = self.imgs[item]
        file_size = os.path.getsize(image_path)

        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            W, H = img.size  # slightly confusing
            bpp = file_size * 8. / (H * W)

            shortest_size_length = min(H, W)
            minimum_scale_factor = 1.0 * self.crop_size / float(shortest_size_length)
            scale_low = max(minimum_scale_factor, self.scale_min)
            scale_high = max(scale_low, self.scale_max)
            scale = np.random.uniform(scale_low, scale_high)

            dynamic_transform = self._transform(scale, H, W)
            transformed = dynamic_transform(img)

        except:
            return None

        return transformed, bpp


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


def get_dataloader(dataset, train_mode=True, json_path=None, shuffle=True, pin_memory=True, batch_size=8,
                   logger=logging.getLogger(__name__), normalize=False, **kwargs):
    """
    A generic data loader
    """

    pin_memory = pin_memory and torch.cuda.is_available()  # Only set pin_memory True if GPU is available
    Dataset = get_dataset(dataset)

    if json_path is None:
        image_dataset = Dataset(logger=logger, train_mode=train_mode, normalize=normalize, **kwargs)
    else:
        image_dataset = Dataset(json_path=json_path, logger=logger, train_mode=train_mode, normalize=normalize, **kwargs)

    return DataLoader(image_dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=NUM_DATASET_WORKERS,
                      collate_fn=exception_collate_fn,
                      pin_memory=pin_memory
                      )


if __name__ == "__main__":
    train_vimeo_loader = get_dataloader("vimeo", train_mode=True, normalize=True)
    print(f"{len(train_vimeo_loader)} train batches")
    val_vimeo_loader = get_dataloader("vimeo", train_mode=False, normalize=True)
    print(f"{len(val_vimeo_loader)} val batches")

    test_kodak_loader = get_dataloader("kodak", normalize=True)
    print(f"{len(test_kodak_loader)} test batches")

    for image, bpp in test_kodak_loader:
        print("Test")