from torch.utils.data.dataset import Dataset

from fling.utils.registry_utils import DATASET_REGISTRY


def get_dataset(args: dict, train: bool) -> Dataset:
    return DATASET_REGISTRY.build(args.data.dataset, args, train)
