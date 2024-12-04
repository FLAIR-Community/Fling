from typing import List, Tuple
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from fling.utils import get_data_transform

from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('domainnet')
class DomainNetDataset(Dataset):
    r"""
    Implementation for DomainNet dataset. 
    You are required to download the original data file from the website manually.
    Details can be viewed in: https://ai.bu.edu/M3SDA/
    """

    def __init__(self, cfg: dict, domain: str, train: bool):
        super(DomainNetDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.transform = get_data_transform(cfg.data.transforms, train=train)
        self.imgs, self.labels = load_domainnet(base_dir=self.cfg.data.data_path, domain=domain, train=train)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> dict:
        img, label = self.imgs[index], self.labels[index]
        if len(img.split()) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)
        img = self.transform(img)

        return {'input': img, 'class_id': label}


def load_domainnet(base_dir: str, domain: str, train: bool) -> Tuple[List, List]:
    r"""
    Overview:
        Function for loading DomainNet dataset according to train signal and base_dir.
    Arguments:
            - base_dir: The base directory of the dataset.
            - domain: Load data from the corresponding domain.
        Returns:
            - Two lists containing image data and labels.
    """
    label_dict = {
        'bird': 0,
        'feather': 1,
        'headphones': 2,
        'ice_cream': 3,
        'teapot': 4,
        'tiger': 5,
        'whale': 6,
        'windmill': 7,
        'wine_glass': 8,
        'zebra': 9
    }
    if train:
        paths, text_labels = np.load('{}DomainNet/split/{}_train.pkl'.format(base_dir, domain), allow_pickle=True)
    else:
        paths, text_labels = np.load('{}DomainNet/split/{}_test.pkl'.format(base_dir, domain), allow_pickle=True)

    labels = [label_dict[text] for text in text_labels]
    imgs = []
    for i in range(len(paths)):
        img_path = os.path.join(base_dir, paths[i])
        img = Image.open(img_path)
        imgs.append(img.copy())

    return imgs, labels
