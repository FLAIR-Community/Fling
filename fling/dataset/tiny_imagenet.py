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

    def __init__(self, cfg, train):
        super(TinyImagenetDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        transform = get_data_transform(cfg.data.transforms, train=train)
        if train:
            self.dataset = ImageFolder(os.path.join(cfg.data.data_path, 'train'), transform=transform)
        else:
            self.dataset = ImageFolder(os.path.join(cfg.data.data_path, 'val'), transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}
