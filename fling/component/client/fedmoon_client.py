import copy

import torch
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedmoon_client')
class FedMOONClient(BaseClient):
    r"""
    Overview:
        This class is the base implementation of client of FedMOON introduced in: Model-Contrastive Federated Learning
     <link https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf link>.
    """

    def __init__(self, *args, **kwargs):
        super(FedMOONClient, self).__init__(*args, **kwargs)
        # The weight of fedmoon loss.
        self.mu = self.args.learn.mu
        # The temperature parameter of fedmoon.
        self.t = self.args.learn.temperature
        # The variable to store the global model.
        self.glob_model = None
        # The variable to store the previous models.
        self.prev_models = []
        # The max length of prev_models
        self.queue_len = self.args.learn.queue_len

    def _store_prev_model(self, model: nn.Module) -> None:
        r"""
        Overview:
            Store the prev model for fedmoon loss calculation.
        """
        if len(self.prev_models) >= self.queue_len:
            self.prev_models.pop(0)
        self.prev_models.append(copy.deepcopy(model))

    def _store_global_model(self, model: nn.Module) -> None:
        r"""
        Overview:
            Store the global model for fedmoon loss calculation.
        """
        self.glob_model = copy.deepcopy(model)

    def train_step(self, batch_data, criterion, monitor, optimizer):
        r"""
        Overview:
            Training step. The loss of fedmoon should be added to the original loss.
        """
        batch_x, batch_y = batch_data['x'], batch_data['y']
        z, o = self.model(batch_x, mode='compute-feature-logit')
        main_loss = criterion(o, batch_y)
        # Calculate fedmoon loss.
        cos = nn.CosineSimilarity(dim=-1)
        self.glob_model.to(self.device)
        with torch.no_grad():
            z_glob, _ = self.glob_model(batch_x, mode='compute-feature-logit')
        z_i = cos(z, z_glob)
        logits = z_i.reshape(-1, 1)
        for prev_model in self.prev_models:
            prev_model.to(self.device)
            with torch.no_grad():
                z_prev, _ = prev_model(batch_x, mode='compute-feature-logit')
            nega = cos(z, z_prev)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.t
        labels = torch.zeros(batch_x.size(0)).to(self.device).long()
        fedmoon_loss = criterion(logits, labels)
        # Add the main loss and fedmoon loss together.
        loss = main_loss + self.mu * fedmoon_loss

        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                'train_acc': torch.mean((y_pred == batch_y).float()).item(),
                'main_loss': main_loss.item(),
                'fedmoon_loss': self.mu * fedmoon_loss.item(),
                'total_loss': loss.item(),
            },
            weight=batch_y.shape[0]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for prev_model in self.prev_models:
            prev_model.to('cpu')

    def train(self, lr, device=None):
        r"""
        Overview:
            Training function. The global model and prev model should be stored.
        """
        self._store_global_model(self.model)
        mean_monitor_variables = super(FedMOONClient, self).train(lr=lr, device=device)
        # Reset the global model to save memory.
        del self.glob_model
        # Store the current model as prev model in next round
        self._store_prev_model(self.model)
        return mean_monitor_variables
