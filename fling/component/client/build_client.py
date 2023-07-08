from typing import Iterable

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.component.client import ClientTemplate


def get_client(train_dataset: Iterable, test_dataset: Iterable, args: dict, client_id: int) -> ClientTemplate:
    return CLIENT_REGISTRY.build(args.client.name, args, train_dataset, test_dataset, client_id)
