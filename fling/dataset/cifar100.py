from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('cifar100')
class CIFAR100Dataset(Dataset):
    r"""
        Implementation for CIFAR100 dataset. Details can be viewed in: https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, cfg: dict, train: bool):
        super(CIFAR100Dataset, self).__init__()
        self.train = train
        self.cfg = cfg
        transform = get_data_transform(cfg.data.transforms, train=train)
        self.dataset = CIFAR100(cfg.data.data_path, train=train, transform=transform, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> int:
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}
