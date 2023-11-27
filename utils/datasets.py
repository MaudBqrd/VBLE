import logging
import numpy as np
from torch.utils.data import Dataset
import os
from os.path import join
import skimage.io as skio
from torchvision import transforms, datasets
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, dataset_root: str, transform=None):
        """
        Dataset class taking images from dataset_root.

        Parameters
        ----------
        dataset_root: path to the dataset
        transform: transform function to apply to each image (contains image normalization)
        """
        self.samples = os.listdir(dataset_root)
        self.samples.sort()
        self.transform = transform
        self.dataset_root = dataset_root

    def __getitem__(self, index: int):
        img = self.open_image(join(self.dataset_root, self.samples[index]))
        if self.transform is not None:
            return self.transform(img).float()
        return img.float()

    def __len__(self):
        return len(self.samples)

    @classmethod
    def open_image(cls, img_path: str):
        ext = img_path.split('.')[-1]
        if ext == 'npy':
            img = np.load(img_path)
        elif ext in ['tiff', 'tif']:
            img = skio.imread(img_path, plugin="tifffile")
        elif ext in ["png", "jpg", "jpeg"]:
            img = np.array(Image.open(img_path))
        else:
            logger = logging.getLogger('logfile')
            logger.error(f"Cannot open {ext} images")
            raise NotImplementedError
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        return img


def load_dataset(dataset: str, dataset_root: str):

    if dataset == 'mnist':

        transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
        test_dataset = datasets.MNIST(dataset_root, train=False, download=False, transform=transform)

    elif dataset == 'celeba':
        ### There is a issue with CelebA on PyTorch code, see https://github.com/pytorch/vision/issues/2262 ###
        transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(128), transforms.Resize(64)])

        test_dataset = datasets.CelebA(dataset_root, split='test', download=False, transform=transform)

    elif dataset == "cifar10":

        transform = transforms.Compose([transforms.ToTensor()])

        test_dataset = datasets.CIFAR10(dataset_root, train=False, download=False, transform=transform)

    else:
        transform_test = transforms.Compose([transforms.ToTensor()])
        test_dataset = ImageDataset(join(dataset_root, dataset), transform=transform_test)

    return test_dataset
