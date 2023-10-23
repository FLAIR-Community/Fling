import copy

import torch
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.utils import get_model_difference
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedprox_client')
class FedProxClient(BaseClient):
    r"""
    Overview:
        This class is the base implementation of client of FedProx introduced in: Federated Optimization in Heterogeneous
    Networks <link https://arxiv.org/pdf/1812.06127.pdf link>.
    """

    def __init__(self, *args, **kwargs):
        super(FedProxClient, self).__init__(*args, **kwargs)
        # The weight of fedprox loss.
        self.mu = self.args.learn.mu
        # The variable to store the global model w0.
        self.glob_model = None

    def _copy_global_model(self, model: nn.Module) -> None:
        r"""
        Overview:
            Copy the current model for fedprox loss calculation.
        """
        model = copy.deepcopy(model.to(self.device))
        parameters = dict(model.to(self.device).named_parameters())
        for key, params in parameters.items():
            params.requires_grad = False
        self.glob_model = parameters

    def train_step(self, batch_data, criterion, monitor, optimizer):
        r"""
        Overview:
            Training step. The loss of fedprox should be added to the original loss.
        """
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        main_loss = criterion(o, batch_y)
        # Calculate fedprox loss.
        fedprox_loss = get_model_difference(dict(self.model.named_parameters()), self.glob_model)
        # Add the main loss and fedprox loss together.
        loss = main_loss + self.mu * fedprox_loss

        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                'train_acc': torch.mean((y_pred == batch_y).float()).item(),
                'main_loss': main_loss.item(),
                'fedprox_loss': fedprox_loss.item(),
                'total_loss': loss.item(),
            },
            weight=batch_y.shape[0]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, lr, device=None, train_args=None):
        r"""
        Overview:
            Training function. The global model should be updated.
        """
        self._copy_global_model(self.model)
        mean_monitor_variables = super(FedProxClient, self).train(lr=lr, device=device, train_args=train_args)
        # Reset the global model to save memory.
        self.glob_model = None
        return mean_monitor_variables

    def finetune(self, lr, finetune_args, device=None, finetune_eps=None, override=False):
        r"""
        Overview:
            Finetune function. The global model should be updated.
        """
        self._copy_global_model(self.model)
        info = super(FedProxClient, self).finetune(lr, finetune_args, device, finetune_eps, override)
        # Reset the global model to save memory.
        self.glob_model = None
        return info
