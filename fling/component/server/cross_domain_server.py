import torch
from torch.utils.data import DataLoader

from fling.utils.registry_utils import SERVER_REGISTRY
from .base_server import BaseServer


@SERVER_REGISTRY.register('cross_domain_server')
class CrossDomainServer(BaseServer):
    """
    Overview:
    This class serves as the base implementation of a server for FL in cross-domain scenarios.
    It is inherited from the `BaseServer` class.
    For the definitions and descriptions of the basic functions `test`,
    please refer to the comments in the `BaseServer` class.
    """

    def __init__(self, args: dict, test_dataset: dict):
        super(CrossDomainServer, self).__init__(args, test_dataset)
        self.domains = args.data.domains.split(',')
        self.test_loader = {}
        for domain in self.domains:
            self.test_loader[domain] = DataLoader(
                test_dataset[domain][0], batch_size=args.learn.batch_size, shuffle=True
            )

    def test_step(self, model, batch_data, domain, criterion, monitor):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                f'{domain}_test_acc': torch.mean((y_pred == batch_y).float()).item(),
                f'{domain}_test_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )
