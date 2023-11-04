import os
import pickle

from torch.utils.data import Dataset
from PIL import Image

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('mini_imagenet')
class MiniImagenetDataset(Dataset):
    """
        Implementation for Mini-Imagenet dataset. Details can be viewed in: \
        https://github.com/yaoyao-liu/mini-imagenet-tools#about-mini-imagenet
    """

    default_augmentation = dict(
        horizontal_flip=dict(p=0.5),
        random_rotation=dict(degree=15),
    )

    def __init__(self, cfg: dict, train: bool):
        super(MiniImagenetDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.transform = get_data_transform(cfg.data.transforms, train=train)

        file_name = 'train_dataset.pkl' if train else 'val_dataset.pkl'
        with open(os.path.join(cfg.data.data_path, file_name), 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict:
        orig_img = Image.fromarray(self.dataset[item][0])
        return {'input': self.transform(orig_img) / 255., 'class_id': self.dataset[item][1]}
