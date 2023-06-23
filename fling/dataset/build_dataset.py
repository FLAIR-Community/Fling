from typing import Iterable

from fling.utils.registry_utils import DATASET_REGISTRY


def get_dataset(args: dict, train: bool) -> Iterable:
    return DATASET_REGISTRY.build(args.data.dataset, args, train)
