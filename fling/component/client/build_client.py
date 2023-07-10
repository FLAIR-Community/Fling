from typing import Iterable

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.component.client import ClientTemplate


def get_client(args: dict, client_id: int, train_dataset: Iterable, test_dataset: Iterable = None) -> ClientTemplate:
    return CLIENT_REGISTRY.build(args.client.name, args, client_id, train_dataset, test_dataset)
