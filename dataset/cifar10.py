from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from utils import get_data_transform
from utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY('cifar10')
class CIFAR10Dataset(Dataset):
    def __init__(self, cfg, train):
        super(CIFAR10Dataset, self).__init__()
        self.train = train
        self.cfg = cfg
        transform = get_data_transform(cfg.data.transform, train=train)
        self.dataset = CIFAR10(cfg.data.data_path, train=train, transform=transform, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
