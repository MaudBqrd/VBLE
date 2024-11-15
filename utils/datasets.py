import logging
import numpy as np
from torch.utils.data import Dataset
import os
from os.path import join
from torchvision import transforms, datasets
from PIL import Image

class ImageDataset(Dataset):

    def __init__(self, target_root: str = None, image_root: str = None, transform=(None, None)):
        """
        Parameters
        ----------
        target_root: path to target images
        image_root: path to degraded images
        transform: (transform to target images, transform to degraded images) transform functions to apply to each image (does not contain image normalization)
        """
        
        if image_root is not None:
            self.image_root = image_root
            self.samples_y = os.listdir(self.image_root)
            self.samples_y.sort()
        else:
            self.image_root = None
        if target_root is not None:
            self.target_root = target_root
            self.samples_gt = os.listdir(self.target_root)
            self.samples_gt.sort()
        else:
            self.target_root = None


        self.transform = transform[0]
        if transform[1] is None:
            self.transform_y = self.transform
        else:
            self.transform_y = transform[1]
        

    def __getitem__(self, index):
        if self.target_root is not None:
            img_target = self.open_image(join(self.target_root, self.samples_gt[index]))
        else:
            img_target = None
        if self.image_root is not None:
            img_degraded = self.open_image(join(self.image_root, self.samples_y[index]))
        else:
            img_degraded = None
        
        if self.transform is not None:
            if img_target is not None:
                img_target = self.transform(img_target).float()
            if img_degraded is not None:
                img_degraded = self.transform_y(img_degraded).float()

        return img_target, img_degraded

    def __len__(self):
        if self.target_root is not None:
            return len(self.samples_gt)
        else:
            return len(self.samples_y)

    @classmethod
    def open_image(cls, img_path):
        ext = img_path.split('.')[-1]
        if ext == 'npy':
            img = np.load(img_path).astype(float)
        elif ext in ["png", "jpg", "jpeg", 'tif']:
            img = np.array(Image.open(img_path)).astype(float)
        else:
            logging.error(f"Cannot open {ext} images")
            raise NotImplementedError
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        return img


def load_dataset(target_image_root: str, degraded_image_root: str = None, scale_factor: int = 1, patch_size: int = None, n_bits: int = 8):
    """
    Args:
    target_image_root: path to target images, if given
    dataset_name: name of the dataset. Not required in general, required for CelebA, MNIST and Cifar10 datasets for specific processing
    degraded_image_root: path to degraded images, if given
    scale_factor: scale factor of the super resolution problem (=1 if no super resolution is performed)
    patch_size: if given, center crop the input images to the given patch size
    n_bits: number of bits the images are coded on. For normalization purposes
    """
        
    transform_list = [transforms.ToTensor()]
    transform_list_y = [transforms.ToTensor()]
    if patch_size is not None:
        transform_list += [transforms.CenterCrop(patch_size)]
        transform_list_y += [transforms.CenterCrop(patch_size // scale_factor)]     

    transform = transforms.Compose(transform_list + [transforms.Normalize(mean=[0], std=[2 ** n_bits - 1])])
    transform_y = transforms.Compose(transform_list_y + [transforms.Normalize(mean=[0], std=[2 ** n_bits - 1])])

    test_dataset = ImageDataset(target_image_root, degraded_image_root, transform=(transform, transform_y))
    
    
    return test_dataset
           

        
        