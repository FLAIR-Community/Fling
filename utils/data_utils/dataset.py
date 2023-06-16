import os

import numpy as np
from torchvision import datasets, transforms


class DatasetConstructor:

    def __init__(self, args):
        self.dataset = args.dataset.lower()
        self.path = args.data_path
        self.resize = args.resize

    def get_dataset(self, train=True):
        path = self.path if self.path is not None else './data/' + self.dataset
        if self.dataset == 'mnist':
            if self.resize > 0:
                transform = transforms.Compose([
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            return datasets.MNIST(path, train=train, download=True, transform=transform)

        elif self.dataset == 'cifar10':
            if self.resize > 0:
                transform = transforms.Compose([
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            return datasets.CIFAR10(os.path.join(path, 'CIFAR10'), train=train, download=True, transform=transform)

        elif self.dataset == 'fashion_mnist':
            if self.resize > 0:
                transform = transforms.Compose([
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            return datasets.FashionMNIST(path, train=train, download=True, transform=transform)

        elif self.dataset == 'imagenet-tiny':
            if train:
                folder = 'train'
            else:
                folder = 'val'
            transform = []
            new_size = 64 if self.resize < 0 else self.resize
            if train:
                transform.append(transforms.RandomHorizontalFlip())
                transform.append(transforms.RandomResizedCrop(new_size, scale=(0.33, 1)))
            else:
                transform.append(transforms.Resize(new_size))
            transform.append(transforms.ToTensor())
            transform = transforms.Compose(transform)
            return datasets.ImageFolder(root=os.path.join(path, 'tiny-imagenet-200', folder), transform=transform)

        elif self.dataset == 'shakespear':
            data_dir = 'data/shakespear'
            if train:
                return np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
            else:
                return np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        elif self.dataset == 'openwebtext':
            data_dir = path
            if train:
                return np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')[:int(1e8)]
            else:
                return np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        else:
            raise ValueError(f'Dataset: {self.dataset} not implemented.')
