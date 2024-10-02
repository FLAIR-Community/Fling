import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.utils import get_model_difference
from fling.utils import get_optimizer, VariableMonitor, get_weights
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedpart_client')
class FedPartClient(BaseClient):
    r"""
    Overview:
        This class is the base implementation of client of FedPart introduced in: 
        Why Go Full? Elevating Federated Learning Through Partial Network Updates 
        <link https:// link>.
    """

    def train(self, lr, device=None, train_args=None):
        """
        Local training.
        """
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.train()
        self.model.to(self.device)

        # Set optimizer, loss function.
        if train_args is None:
            weights = self.model.parameters()
        else:
            weights = get_weights(self.model, parameter_args=train_args,
                                  include_non_param=False, prefix_mode=True)
        op = get_optimizer(weights=weights, **self.args.learn.optimizer)

        # Set the loss function.
        criterion = nn.CrossEntropyLoss()

        monitor = VariableMonitor()

        # Main training loop.
        for epoch in range(self.args.learn.local_eps):
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.train_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=op)

        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        if device is not None:
            self.device = device_bak

        return mean_monitor_variables
    
    def finetune(self, lr, finetune_args, device=None, finetune_eps=None, override=False):
        """
        Finetune function. In this function, the local model will not be changed, but will return the finetune results.
        """
        # Back-up variables.
        if device is not None:
            device_bak = self.device
            self.device = device
        if not override:
            model_bak = copy.deepcopy(self.model)

        # Get default ``finetune_eps``.
        if finetune_eps is None:
            finetune_eps = self.args.learn.local_eps

        self.model.train()
        self.model.to(self.device)

        # Get weights to be fine-tuned.
        # For calculating train loss and train acc.
        weights = get_weights(self.model, parameter_args=finetune_args,
                              include_non_param=False, prefix_mode=True)

        # Get optimizer and loss.
        op = get_optimizer(weights=weights, **self.args.learn.optimizer)
        criterion = nn.CrossEntropyLoss()

        # Main loop.
        info = []
        for epoch in range(finetune_eps):
            self.model.train()
            self.model.to(self.device)
            monitor = VariableMonitor()
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.finetune_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=op)

            # Test model every epoch.
            mean_monitor_variables = monitor.variable_mean()
            mean_monitor_variables.update(self.test())
            info.append(mean_monitor_variables)

        # Retrieve the back-up variables.
        if not override:
            self.model = model_bak
        else:
            # Put the model to cpu after training to save GPU memory.
            self.model.to('cpu')
        if device is not None:
            self.device = device_bak

        return info