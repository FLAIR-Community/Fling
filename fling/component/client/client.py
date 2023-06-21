import copy
import random

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from fling.model import get_model
from fling.utils import get_optimizer, VariableMonitor, get_finetune_parameters
from fling.utils.registry_utils import CLIENT_REGISTRY


@CLIENT_REGISTRY.register('base_client')
class Client:
    """
    Overview:
    This class is the base implementation of client in Federated Learning. Typically, a client need to have these functions.
    ``train``: A client need to define the local training process.
    ``test``: A client need to define how to test the local model given a dataset.
    ``finetune``: A client need to define how to finetune the local model (usually used in Personalized Federated Learning)
    If users want to define a new client class, it is recommended to inherit this class.
    """

    def __init__(self, args, train_dataset, client_id):
        """
        Initializing train dataset, test dataset(for personalized settings), constructing model and other configurations.
        """
        test_frac = args.client.test_frac
        self.args = args
        # If test_frac > 0, it means that a fraction of the given dataset will be separated for testing.
        if test_frac == 0:
            # ``self.sample_num`` refers to the number of local training number.
            self.sample_num = len(train_dataset)
            self.train_dataloader = DataLoader(train_dataset, batch_size=args.learn.batch_size, shuffle=True)
        else:
            # Separate a fraction of ``train_dataset`` for testing.
            real_train = copy.deepcopy(train_dataset)
            real_test = copy.deepcopy(train_dataset)
            # Get the indexes of train dataset.
            indexes = real_train.indexes
            random.shuffle(indexes)
            # Randomly sampling a part to be test dataset.
            train_index = indexes[:int((1 - test_frac) * len(train_dataset))]
            test_index = indexes[int((1 - test_frac) * len(train_dataset)):]
            real_train.indexes = train_index
            real_test.indexes = test_index
            # ``self.sample_num`` refers to the number of local training number.
            self.sample_num = len(real_train)

            self.train_dataloader = DataLoader(real_train, batch_size=args.learn.batch_size, shuffle=True)
            self.test_dataloader = DataLoader(real_test, batch_size=args.learn.batch_size, shuffle=True)

        # Model construction.
        self.model = get_model(args)
        self.device = args.learn.device
        # Specify a unique client id.
        self.client_id = client_id
        # This attribute will not be set until ``self.set_fed_keys(self, keys)`` is called.
        # Only weights in ``self.fed_keys`` will be collaboratively trained using Federated Learning.
        self.fed_keys = []

    def set_fed_keys(self, keys):
        """
        Set the attribute ``self.fed_keys``.
        Only weights in ``self.fed_keys`` will be collaboratively trained using Federated Learning.
        """
        self.fed_keys = keys

    def update_model(self, dic):
        """
        Using the ``dic`` to update the state_dict of the local model of this client.
        For keys not existed in ``dic``, the value will be retained.
        """
        dic = copy.deepcopy(dic)
        state_dict = self.model.state_dict()
        state_dict.update(dic)

        self.model.load_state_dict(state_dict)

    def train(self, lr):
        """
        Local training.
        """
        self.model.train()
        self.model.to(self.device)

        # Set optimizer, loss function.
        weights = self.model.parameters()
        op = get_optimizer(
            name=self.args.learn.optimizer.name, lr=lr, momentum=self.args.learn.optimizer.momentum, weights=weights
        )
        criterion = nn.CrossEntropyLoss()

        monitor = VariableMonitor(['acc', 'loss'])

        # Main training loop.
        for epoch in range(self.args.learn.local_eps):
            for _, (batch_x, batch_y) in enumerate(self.train_dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                # Update total sample number.
                o = self.model(batch_x)
                loss = criterion(o, batch_y)
                y_pred = torch.argmax(o, dim=-1)

                monitor.append(
                    {
                        'acc': torch.mean((y_pred == batch_y).float).item(),
                        'loss': loss.item()
                    }, weight=batch_y.shape[0]
                )
                op.zero_grad()
                loss.backward()
                op.step()

        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        return mean_monitor_variables

    def finetune(self, lr, finetune_args, finetune_eps=None):
        """
        Finetune function. In this function, the local model will not be changed, but will return the finetune results.
        """
        model_bak = copy.deepcopy(self.model)
        self.model.train()
        self.model.to(self.device)

        info = []

        # Get weights to be finetuned.
        # For calculating train loss and train acc.
        weights = get_finetune_parameters(self.model, finetune_args=self.args.learn.finetune_parameters)

        # Get optimizer and loss.
        op = get_optimizer(name=self.args.learn.optimizer, lr=lr, momentum=self.args.learn.momentum, weights=weights)
        criterion = nn.CrossEntropyLoss()

        # Main loop.
        if finetune_eps is None:
            finetune_eps = self.args.learn.loc_epoch
        for epoch in range(finetune_eps):
            self.model.train()
            self.model.to(self.device)
            train_variable_monitor = VariableMonitor(['train_acc', 'test_acc'])
            for _, (batch_x, batch_y) in enumerate(self.train_dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                o = self.model(batch_x)
                loss = criterion(o, batch_y)
                y_pred = torch.argmax(o, dim=-1)
                op.zero_grad()
                loss.backward()
                op.step()
                train_variable_monitor.append(
                    {
                        'train_acc': torch.mean((y_pred == batch_y).float).item(),
                        'train_loss': loss.item()
                    },
                    weight=batch_y.shape[0]
                )

            # Test model every epoch.
            mean_monitor_variables = train_variable_monitor.variable_mean()
            mean_monitor_variables.update(self.test)
            info.append(mean_monitor_variables)

        self.model = model_bak

        return info

    def test(self):
        """
        Test model.
        """
        self.model.eval()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor(['acc', 'loss'])

        # Main test loop.
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(self.test_dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                o = self.model(batch_x)
                loss = criterion(o, batch_y)
                y_pred = torch.argmax(o, dim=-1)

                monitor.append(
                    {
                        'acc': torch.mean((y_pred == batch_y).float).item(),
                        'loss': loss.item()
                    }, weight=batch_y.shape[0]
                )

        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        return mean_monitor_variables

    def get_state_dict(self, keys):
        """
        Get the state dict of local model.
        """
        state_dict = self.model.state_dict()
        return {k: state_dict[k] for k in keys}
