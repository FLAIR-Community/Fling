import copy

import torch
import torch.nn.functional as F

from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('pfedsd_client')
class pFedSDClient(BaseClient):
    """
    Overview:
        This class is the base implementation of pFedSD, which is introduced in:
        Personalized Edge Intelligence via Federated Self-Knowledge Distillation.
        <link https://ieeexplore.ieee.org/abstract/document/9964434 link>
    """

    def __init__(self, args, client_id, train_dataset, test_dataset=None):
        """
        Initializing train dataset, test dataset(for personalized settings).
        """
        super(pFedSDClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.local_model_pre = None  # record the local model in the previous round, used for distillation in pfedsd

    def train(self, lr, device=None, train_args=None):
        """
        Local training.
        """
        mean_monitor_variables = super().train(lr, device, train_args=train_args)
        # record the local model in this round
        self.local_model_pre = copy.deepcopy(self.model)

        return mean_monitor_variables

    def train_step(self, batch_data, criterion, monitor, optimizer):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        # execute self distillation in pfedsd
        if self.local_model_pre is not None:
            self.local_model_pre.to(self.device)
            with torch.no_grad():
                o_pre = self.local_model_pre(batch_x)
            lamda = self.args.learn.lamda
            tau = self.args.learn.tau
            q_v = F.softmax(o_pre / tau, dim=-1)
            q_w = F.log_softmax(o / tau, dim=-1)
            loss += lamda * tau ** 2 * F.kl_div(q_w, q_v, reduction='batchmean')

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
