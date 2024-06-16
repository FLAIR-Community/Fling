from torch.utils.data.dataset import Dataset

from fling.utils.registry_utils import DATASET_REGISTRY


def get_dataset(args: dict, train: bool, **kwargs) -> Dataset:
    return DATASET_REGISTRY.build(args.data.dataset, cfg=args, train=train, **kwargs)
