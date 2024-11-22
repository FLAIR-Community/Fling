import copy
import random

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('cross_domain_client')
class CrossDomainClient(BaseClient):
    """
    Overview:
    This class serves as the base implementation of a client for federated learning in cross-domain scenarios.
    It is inherited from the `BaseClient` class.
    For the definitions and descriptions of the basic functions `train`, `test`, and `finetune`, 
    please refer to the comments in the `BaseClient` class.
    If users want to define a new client class for cross-domain scenario, it is recommended to inherit this class.
    """

    def __init__(self, args: dict, domain: str, client_id: int, train_dataset: Dataset, test_dataset: Dataset = None):
        """
        In addition to the initialization in the `BaseClient` class, 
        it is also necessary to define the source domain information for the client's data.
        """
        super(CrossDomainClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.domain = domain

    def train_step(self, batch_data, criterion, monitor, optimizer):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                f'{self.domain}_{self.client_id}_train_acc': torch.mean((y_pred == batch_y.long()).float()).item(),
                f'{self.domain}_{self.client_id}_train_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def finetune_step(self, batch_data, criterion, monitor, optimizer):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                f'{self.domain}_{self.client_id}_train_acc': torch.mean((y_pred == batch_y.long()).float()).item(),
                f'{self.domain}_{self.client_id}_train_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_step(self, batch_data, criterion, monitor):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                f'{self.domain}_{self.client_id}_test_acc': torch.mean((y_pred == batch_y.long()).float()).item(),
                f'{self.domain}_{self.client_id}_test_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )
