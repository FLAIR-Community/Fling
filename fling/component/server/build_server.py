from torch.utils.data.dataset import Dataset

from fling.utils.registry_utils import SERVER_REGISTRY
from fling.component.server import ServerTemplate


def get_server(args: dict, test_dataset: Dataset, **kwargs) -> ServerTemplate:
    return SERVER_REGISTRY.build(args.server.name, args=args, test_dataset=test_dataset, **kwargs)
