import copy
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient
from fling.utils.utils import weight_flatten

@CLIENT_REGISTRY.register('fedamp_client')
class FedAMPClient(BaseClient):
    """
    Overview:
        This class is the base implementation of client in 'Bold but Cautious: Unlocking the Potential of Personalized
        Federated Learning through Cautiously Aggressive Collaboration' (FedCAC).
    """

    def __init__(self, args, client_id, train_dataset, test_dataset=None):
        """
        Initializing train dataset, test dataset(for personalized settings).
        """
        super(FedAMPClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.client_u = copy.deepcopy(self.model)

    def FedAMP_Loss_client(self):
        params = weight_flatten(self.model)
        params_ = weight_flatten(self.client_u)
        sub = params - params_
        result = self.args.learn.lamda / (2 * self.args.learn.alphaK) * torch.dot(sub, sub)
        return result

    def train_step(self, batch_data, criterion, monitor, optimizer):
        self.client_u.to(self.device)

        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
        loss = loss + self.FedAMP_Loss_client()
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                'train_acc': torch.mean((y_pred == batch_y).float()).item(),
                'train_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.client_u.to('cpu')