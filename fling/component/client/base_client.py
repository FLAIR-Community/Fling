import copy
import random

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from fling.utils import get_optimizer, VariableMonitor, get_weights
from fling.utils.registry_utils import CLIENT_REGISTRY
from .client_template import ClientTemplate


@CLIENT_REGISTRY.register('base_client')
class BaseClient(ClientTemplate):
    """
    Overview:
    This class is the base implementation of client in Federated Learning.
    Typically, a client need to have these functions.
    ``train``: A client need to define the local training process.
    ``test``: A client need to define how to test the local model given a dataset.
    ``finetune``: A client need to define how to finetune the local model (usually used in Personalized FL)
    If users want to define a new client class, it is recommended to inherit this class.
    """

    def __init__(self, args: dict, client_id: int, train_dataset: Dataset, test_dataset: Dataset = None):
        """
        Overview:
            Initializing train dataset, test dataset(for personalized settings).
        Arguments:
            - args: dict type arguments.
            - train_dataset: private dataset for training
            - test_dataset: private dataset for testing (Optional)
            - client_id: unique id for this client.
        Returns:
            - None
        """
        super(BaseClient, self).__init__(args, client_id, train_dataset, test_dataset)
        val_frac = args.client.val_frac
        # If val_frac > 0, it means that a fraction of the given dataset will be separated for validating.
        if val_frac == 0:
            # ``self.sample_num`` refers to the number of local training number.
            self.sample_num = len(train_dataset)
            self.train_dataloader = DataLoader(train_dataset, batch_size=args.learn.batch_size, shuffle=True)
        else:
            # Separate a fraction of ``train_dataset`` for validating.
            real_train = copy.deepcopy(train_dataset)
            real_test = copy.deepcopy(train_dataset)
            # Get the indexes of train dataset.
            indexes = real_train.indexes
            random.shuffle(indexes)
            # Randomly sampling a part to be test dataset.
            train_index = indexes[:int((1 - val_frac) * len(train_dataset))]
            test_index = indexes[int((1 - val_frac) * len(train_dataset)):]
            real_train.indexes = train_index
            real_test.indexes = test_index
            # ``self.sample_num`` refers to the number of local training number.
            self.sample_num = len(real_train)

            self.train_dataloader = DataLoader(real_train, batch_size=args.learn.batch_size, shuffle=True)
            self.val_dataloader = DataLoader(real_test, batch_size=args.learn.batch_size, shuffle=True)

        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, batch_size=args.learn.batch_size, shuffle=True)

    def train_step(self, batch_data, criterion, monitor, optimizer):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
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

    def finetune_step(self, batch_data, criterion, monitor, optimizer):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
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

    def test_step(self, batch_data, criterion, monitor):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = self.model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                'test_acc': torch.mean((y_pred == batch_y).float()).item(),
                'test_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )

    def preprocess_data(self, data):
        return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}

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
            weights = get_weights(self.model, parameter_args=train_args)
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
        weights = get_weights(self.model, parameter_args=finetune_args)

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

    def test(self):
        """
        Test model.
        """
        self.model.eval()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()

        # Main test loop.
        with torch.no_grad():
            for _, data in enumerate(self.test_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.test_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor)

        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        return mean_monitor_variables
