from typing import Iterable

from fling.utils.registry_utils import SERVER_REGISTRY
from fling.component.server import ServerTemplate


def get_server(args: dict, test_dataset: Iterable) -> ServerTemplate:
    return SERVER_REGISTRY.build(args.server.name, args, test_dataset)
