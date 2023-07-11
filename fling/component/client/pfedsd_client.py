import copy
import random

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from fling.utils import get_optimizer, VariableMonitor, get_finetune_parameters
from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient
import torch.nn.functional as F


@CLIENT_REGISTRY.register('pfedsd_client')
class pFedSDClient(BaseClient):
    """
    Overview:
        This class is the base implementation of client in Federated Learning + lora.
    """

    def __init__(self, args, client_id, train_dataset, test_dataset=None):
        """
        Initializing train dataset, test dataset(for personalized settings).
        """
        super(pFedSDClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.local_model_pre = None # record the local model in the previous round, used for distillation in pfedsd

    def train(self, lr, device=None):
        """
        Local training.
        """
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.train()
        self.model.to(self.device)

        # Set optimizer, loss function.
        weights = self.model.parameters()
        op = get_optimizer(
            name=self.args.learn.optimizer.name,
            lr=lr,
            momentum=self.args.learn.optimizer.momentum,
            weights=weights
        )

        criterion = nn.CrossEntropyLoss()

        monitor = VariableMonitor()

        # Main training loop for lora
        for epoch in range(self.args.learn.local_eps):
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.train_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=op)

        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()



        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        # record the local model in this round
        self.local_model_pre = copy.deepcopy(self.model)

        if device is not None:
            self.device = device_bak

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