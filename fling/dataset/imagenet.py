from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('imagenet')
class ImagenetDataset(Dataset):
    """
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
        self.transform = get_data_transform(cfg.data.transforms, train=train)
        split = 'train' if train else 'val'
        self.dataset = ImageNet(root=cfg.data.data_path, split=split, transform=self.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict:
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}
