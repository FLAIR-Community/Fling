import copy
import random

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from fling.utils import get_optimizer, VariableMonitor, get_finetune_parameters
from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedlora_client')
class FedLoraClient(BaseClient):
    """
    Overview:
        This class is the base implementation of client in Federated Learning + lora.
    """

    def __init__(self, args, train_dataset, test_dataset, client_id):
        """
        Initializing train dataset, test dataset(for personalized settings).
        """
        super(FedLoraClient, self).__init__(args, train_dataset, test_dataset, client_id)

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
        weights = dict(self.model.named_parameters())
        lora_weights, non_lora_weights = [], []
        for k in weights.keys():
            if 'lora_A' in k or 'lora_B' in k:
                lora_weights.append(weights[k])
            else:
                non_lora_weights.append(weights[k])
        lora_op = get_optimizer(
            name=self.args.learn.optimizer.name,
            lr=lr,
            momentum=self.args.learn.optimizer.momentum,
            weights=lora_weights
        )

        non_lora_op = get_optimizer(
            name=self.args.learn.optimizer.name,
            lr=lr,
            momentum=self.args.learn.optimizer.momentum,
            weights=non_lora_weights
        )

        criterion = nn.CrossEntropyLoss()

        monitor = VariableMonitor(['train_acc', 'train_loss'])

        # Main training loop for lora
        for epoch in range(self.args.learn.local_lora_eps):
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.train_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=lora_op)

        monitor = VariableMonitor(['train_acc', 'train_loss'])
        # Main training loop for non-lora
        for epoch in range(self.args.learn.local_eps - self.args.learn.local_lora_eps):
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.train_step(
                    batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=non_lora_op
                )

        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        if device is not None:
            self.device = device_bak

        return mean_monitor_variables
