import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('tiny_imagenet')
class TinyImagenetDataset(Dataset):
    r"""
        Implementation for Tiny-Imagenet dataset. Details can be viewed in:
        http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """

    default_augmentation = dict(
        horizontal_flip=dict(p=0.5),
        random_rotation=dict(degree=15),
        Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    )

    def __init__(self, cfg: dict, train: bool):
        super(TinyImagenetDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        transform = get_data_transform(cfg.data.transforms, train=train)
        if train:
            self.dataset = ImageFolder(os.path.join(cfg.data.data_path, 'train'), transform=transform)
        else:
            self.dataset = ImageFolder(os.path.join(cfg.data.data_path, 'val'), transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict:
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}
