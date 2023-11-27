import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_dataset(dataset: str, batch_size: int, dataset_root: str, num_workers: int = 4, shuffle=True, device='cuda'):
    """
    Parameters
    ----------
    dataset: dataset name (mnist, celeba, cifar10 implemented)
    batch_size: batch size for dataloaders
    dataset_root: path to the image folder
    num_workers: number of workers for dataloaders
    shuffle: True to shuffle in dataloader
    device: cuda or cpu
    """

    if dataset == 'mnist':

        transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])

        mnist_dataset = datasets.MNIST(dataset_root, train=True, download=False, transform=transform)
        n_sample_val = len(mnist_dataset) // 10
        indice_val = np.arange(0, n_sample_val)
        indice_train = np.arange(n_sample_val, len(mnist_dataset))

        train_loader = DataLoader(
            Subset(mnist_dataset, indice_train),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device == 'cuda'
        )
        val_loader = DataLoader(
            Subset(mnist_dataset, indice_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device == 'cuda'
        )

    elif dataset == "cifar10":

        transform = transforms.Compose([transforms.ToTensor()])

        cifar_dataset = datasets.CIFAR10(dataset_root, train=True, download=False, transform=transform)
        cifar_dataset_test = datasets.CIFAR10(dataset_root, train=False, download=False, transform=transform)
        n_sample_val = len(cifar_dataset) // 10
        indice_val = np.arange(0, n_sample_val)

        train_loader = DataLoader(
            cifar_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device == 'cuda'
        )
        val_loader = DataLoader(
            Subset(cifar_dataset_test, indice_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device == 'cuda'
        )

    elif dataset == 'celeba':
        ### There is a issue with CelebA on PyTorch code, see https://github.com/pytorch/vision/issues/2262 ###
        transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(128), transforms.Resize(64)])

        train_dataset = datasets.CelebA(dataset_root, split='train', download=False, transform=transform)
        test_dataset = datasets.CelebA(dataset_root, split='valid', download=False, transform=transform)

        n_sample_val = len(train_dataset) // 10
        indice_val = np.arange(0, min(n_sample_val, len(test_dataset)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device == 'cuda'
        )
        val_loader = DataLoader(
            Subset(test_dataset, indice_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device == 'cuda'
        )

    else:
        raise NotImplementedError

    return train_loader, val_loader


