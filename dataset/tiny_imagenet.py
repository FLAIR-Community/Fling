from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils import get_data_transform
from utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY('tiny_imagenet')
class TinyImagenetDataset(Dataset):
    def __init__(self, cfg, train):
        super(TinyImagenetDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        transform = get_data_transform(cfg.data.transform, train=train)
        self.dataset = ImageFolder(cfg.data.data_path, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
