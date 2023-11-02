from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fling.utils.registry_utils import SERVER_REGISTRY
from .base_server import BaseServer


@SERVER_REGISTRY.register('fedmoon_server')
class FedMOONServer(BaseServer):
    r"""
    Overview:
        This class is the base implementation of server of FedMOON introduced in: Model-Contrastive Federated Learning
     <link https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf link>.
    """

    def __init__(self, *args, **kwargs):
        super(FedMOONServer, self).__init__(*args, **kwargs)

    def test_step(self, model, batch_data, criterion, monitor):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        _, o = model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                'test_acc': torch.mean((y_pred == batch_y).float()).item(),
                'test_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )
