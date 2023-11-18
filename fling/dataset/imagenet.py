import os
import pickle

import lmdb
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('imagenet')
class ImagenetDataset(Dataset):
    """
    Overview:
        Implementation for Imagenet dataset. You are required to download the original data file from the website \
    manually. For detailed information, you can refer to the doc of torchvision: \
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/imagenet.html
    """
    default_augmentation = dict(
        random_resized_crop=dict(size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        horizontal_flip=dict(p=0.5),
        Normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.24, 0.225]),
    )

    def __init__(self, cfg: dict, train: bool):
        super(ImagenetDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.use_lmdb = cfg.data.get("use_lmdb", False)
        self.transform = get_data_transform(cfg.data.transforms, train=train)

        if not self.use_lmdb:
            split = 'train' if train else 'val'
            self.dataset = ImageNet(root=cfg.data.data_path, split=split, transform=self.transform)
            self.length = len(self.dataset)
        else:
            if train:
                db_path = os.path.join(cfg.data.data_path, 'train.lmdb')
            else:
                db_path = os.path.join(cfg.data.data_path, 'val.lmdb')
            self.db_path = db_path
            env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, max_readers=16, create=False)
            with env.begin(write=False) as txn:
                self.length = pickle.loads(txn.get(b'__len__'))
                self.keys = pickle.loads(txn.get(b'__keys__'))

    def _open_lmdb(self):
        env = lmdb.open(self.db_path, subdir=False, readonly=True, create=False, max_readers=16, lock=False)
        self.txn = env.begin(buffers=True)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict:
        if not self.use_lmdb:
            return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}
        else:
            if not hasattr(self, 'txn'):
                self._open_lmdb()
            byte_flow = self.txn.get(self.keys[item])
            unpacked = pickle.loads(byte_flow)
            img, label = unpacked[0], unpacked[1]
            return {'input': self.transform(img), 'class_id': label}
