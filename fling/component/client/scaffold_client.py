import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.utils import get_model_difference
from fling.utils import get_optimizer, VariableMonitor, get_weights
from .base_client import BaseClient


@CLIENT_REGISTRY.register('scaffold_client')
class SCAFFOLDClient(BaseClient):
    r"""
    Overview:
        This class is the base implementation of client of Scaffold introduced in: 
        SCAFFOLD: Stochastic Controlled Averaging for Federated Learning 
        <link https://arxiv.org/abs/1910.06378 link>.
    """

    def __init__(self, *args, **kwargs):
        super(SCAFFOLDClient, self).__init__(*args, **kwargs)
        self.c = {k: torch.zeros_like(v.data) for k, v in self.model.named_parameters()}
        self.delta_c = {k: torch.zeros_like(v.data) for k, v in self.model.named_parameters()}
        self.delta_y = {k: torch.zeros_like(v.data) for k, v in self.model.named_parameters()}
        self.server_c = {k: torch.zeros_like(v.data) for k, v in self.model.named_parameters()}
        self.model.to(kwargs['args']['learn']['device'])

    def train(self, lr, device=None, train_args=None):
        """
        Local training.
        """
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.train()
        self.model.to(self.device)
        for k, v in self.model.named_parameters():
            self.c[k] = self.c[k].to(self.device)
            self.delta_c[k] = self.delta_c[k].to(self.device)
            self.delta_y[k] = self.delta_y[k].to(self.device)
            self.server_c[k] = self.server_c[k].to(self.device)

        # Set optimizer, loss function.
        weights = self.model.parameters()
        op = SCAFFOLDOptimizer(params=weights, server_c=self.server_c, client_c=self.c, lr=lr)

        # Set the loss function.
        criterion = nn.CrossEntropyLoss()

        monitor = VariableMonitor()

        # server_weights records the parameters before the optimizer update
        server_weights = {}
        for k, v in self.model.named_parameters():
            server_weights[k] = copy.deepcopy(v.data).to(self.device)

        # Main training loop.
        for epoch in range(self.args.learn.local_eps):
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.train_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=op)

        iter_num_per_epoch = (self.sample_num + self.args.learn.batch_size - 1) // self.args.learn.batch_size
        K = self.args.learn.local_eps * iter_num_per_epoch
        for k, v in self.model.named_parameters():
            new_c = self.c[k] - self.server_c[k] + (server_weights[k] - v.data) / (K * lr)
            self.delta_y[k] = v.data - server_weights[k]
            self.delta_c[k] = new_c - self.c[k]
            self.c[k] = new_c

        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        if device is not None:
            self.device = device_bak

        return mean_monitor_variables

    def update_c(self, dic: dict) -> None:
        r"""
        Overview:
            Update the c_vector of the local model of this client.
            For keys not existed in the argument `dic`, the value will be retained.
        Arguments:
            - dic: dict type parameters for updating local model.
        Returns:
            - None
        """
        self.server_c.update(dic)


class SCAFFOLDOptimizer(Optimizer):

    def __init__(self, params, server_c, client_c, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
        self.server_c = server_c
        self.client_c = client_c

    def step(self):
        for group in self.param_groups:
            for p, c, ci in zip(group['params'], self.server_c.values(), self.client_c.values()):
                if p.grad is None:
                    continue
                grad = p.grad.data + c.data - ci.data
                p.data = p.data - group['lr'] * grad.data
