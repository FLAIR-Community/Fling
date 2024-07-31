from torch.utils.data import Dataset
from torchvision.datasets import EMNIST

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('emnist')
class EMNISTDataset(Dataset):
    r"""
        Implementation for EMNIST dataset. Details can be viewed in: https://www.westernsydney.edu.au/icns/resources/reproducible_research3/publication_support_materials2/emnist
    """

    def __init__(self, cfg: dict, train: bool):
        super(EMNISTDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        transform = get_data_transform(cfg.data.transforms, train=train)
        self.dataset = EMNIST(cfg.data.data_path, split=cfg.data.split, train=train, transform=transform, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict:
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}
