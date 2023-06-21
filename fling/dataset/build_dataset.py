from fling.utils.registry_utils import DATASET_REGISTRY


def get_dataset(args, train):
    return DATASET_REGISTRY.build(args.data.dataset, args, train)
