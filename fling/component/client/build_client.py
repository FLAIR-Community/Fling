from torch.utils.data.dataset import Dataset

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.component.client import ClientTemplate
from fling.component.client import CrossDomainClientTemplate


def get_client(args: dict, client_id: int, train_dataset: Dataset, test_dataset: Dataset = None) -> ClientTemplate:
    return CLIENT_REGISTRY.build(args.client.name, args, client_id, train_dataset, test_dataset)

def get_cross_domain_client(args: dict, domain: str, client_id: int, train_dataset: Dataset, test_dataset: Dataset = None) -> CrossDomainClientTemplate:
    return CLIENT_REGISTRY.build(args.client.name, args, domain, client_id, train_dataset, test_dataset)
