# Overall Framework of Fling

## Fling Components

### Overview

​	In the design for Fling, we extract three main components in the process of federated learning, they are respectively: client, server and group.

- **Client:** A client refers to a computational node. Each client has its own private dataset and use it to update the federated model.
- **Server:** A server refers to the central computational node. A server typically serves for storing a global model, testing the performance of global model.
- **Group:** A group refers to a set consisting of several clients and servers (usually one server and multiple clients). For clients in the same group, they will perform federated learning.

A detailed introduction of these three components is shown as below:

### Client

```python
import copy
from typing import Callable, Iterable
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset

from fling.model import get_model
from fling.utils import VariableMonitor


class ClientTemplate:
    r"""
    Overview:
        Template of client in Federated Learning.
    """

    def __init__(self, args: dict, client_id: int, train_dataset: Dataset, test_dataset: Dataset = None):
        r"""
        Overview:
            Initialization for a client.
        Arguments:
            - args: dict type arguments.
            - train_dataset: private dataset for training
            - test_dataset: private dataset for testing (Optional)
            - client_id: unique id for this client.
        Returns:
            - None
        """
        # Model construction.
        self.args = args
        self.model = get_model(args)
        self.device = args.learn.device
        # Specify a unique client id.
        self.client_id = client_id
        # This attribute will not be set until ``self.set_fed_keys(self, keys)`` is called.
        # Only weights in ``self.fed_keys`` will be collaboratively trained using Federated Learning.
        self.fed_keys = []

    def set_fed_keys(self, keys: Iterable) -> None:
        r"""
        Overview:
            Set `self.fed_dict` to determine which parameters should be aggregated.
        Arguments:
            - keys: sequence that contains the keys of parameters that need to be aggregated.
        Returns:
            - None
        """
        self.fed_keys = list(keys)

    def update_model(self, dic: dict) -> None:
        r"""
        Overview:
            Update the state_dict of the local model of this client.
            For keys not existed in the argument `dic`, the value will be retained.
        Arguments:
            - dic: dict type parameters for updating local model.
        Returns:
            - None
        """
        dic = copy.deepcopy(dic)
        state_dict = self.model.state_dict()
        state_dict.update(dic)

        self.model.load_state_dict(state_dict)

    def get_state_dict(self, keys: Iterable) -> dict:
        r"""
        Overview:
            Get the parameter diction of local model.
        Arguments:
            - keys: sequence that contains the keys of parameters that are acquired.
        Returns:
            - partial_dict: the acquired diction of parameters.
        """
        state_dict = self.model.state_dict()
        partial_dict = {k: state_dict[k] for k in keys}
        return partial_dict

    def train_step(self, batch_data: dict, criterion: Callable, monitor: VariableMonitor, optimizer: Optimizer) -> None:
        r"""
        Overview:
            A step of local training given one data batch.
        Arguments:
            - batch_data: dict type data for updating local model.
            - criterion: loss function.
            - monitor: variable monitor for results generated in each step.
            - optimizer: optimizer for training local model
        Returns:
            - None
        """
        raise NotImplementedError

    def test_step(self, batch_data: dict, criterion: Callable, monitor: VariableMonitor) -> None:
        r"""
        Overview:
            A step of local testing given one data batch.
        Arguments:
            - batch_data: dict type data for testing local model.
            - criterion: loss function.
            - monitor: variable monitor for results generated in each step.
        Returns:
            - None
        """
        raise NotImplementedError

    def preprocess_data(self, data: dict) -> dict:
        r"""
        Overview:
            Pre-process the data batch generated from dataset.
        Arguments:
            - data: raw data generated from dataset.
        Returns:
            - Data after pre-processing.
        """
        raise NotImplementedError

    def train(self, lr: float, device: str, train_args: dict = None) -> dict:
        r"""
        Overview:
            The local training process of a client.
        Arguments:
            - lr: learning rate of the training.
            - device: device for operating this function.
        Returns:
            - A diction containing training results.
        """
        raise NotImplementedError

    def finetune(self, lr: float, finetune_args: dict, device: str, finetune_eps: int, override: bool) -> list:
        r"""
        Overview:
            The local fine-tuning process of a client.
        Arguments:
            - lr: learning rate of the training.
            - finetune_args: arguments for fine-tuning.
            - device: device for operating this function.
            - finetune_eps: epochs for fine-tuning.
            - override: whether to override ``self.model`` using the fine-tuning result.
        Returns:
            - A list of diction containing fine-tuning results.
        """
        raise NotImplementedError

    def test(self) -> dict:
        r"""
        Overview:
            The local testing process of a client.
        Returns:
            - A diction containing testing results.
        """
        raise NotImplementedError
```

### Server

```python
class ServerTemplate:
    r"""
    Overview:
        Template of server in Federated Learning.
    """

    def __init__(self, args: Dict, test_dataset: Dataset):
        r"""
        Overview:
            Initialization for a server.
        Arguments:
            - args: dict type arguments.
            - test_dataset: test dataset.
        Returns:
            - None
        """
        self.args = args
        self.glob_dict = None

        device = args.learn.device
        self.device = device

    def apply_grad(self, grad: Dict, lr: float = 1.) -> None:
        r"""
        Overview:
            Using the averaged gradient to update global model.
        Arguments:
            - grad: dict type gradient.
            - lr: update learning rate.
        Returns:
            - None
        """
        state_dict = self.glob_dict
        for k in grad:
            state_dict[k] = state_dict[k] + lr * grad[k]

    def test_step(self, model: nn.Module, batch_data: Dict, criterion: Callable, monitor: Logger) -> None:
        r"""
        Overview:
            A step of local testing given one data batch.
        Arguments:
            - batch_data: dict type data for testing local model.
            - criterion: loss function.
            - monitor: variable monitor for results generated in each step.
        Returns:
            - None
        """
        raise NotImplementedError

    def preprocess_data(self, data: Dict) -> Dict:
        r"""
        Overview:
            Pre-process the data batch generated from dataset.
        Arguments:
            - data: raw data generated from dataset.
        Returns:
            - Data after pre-processing.
        """
        raise NotImplementedError

    def test(self, model: nn.Module, test_loader: DataLoader = None) -> Dict:
        r"""
        Overview:
            The local testing process of a client.
        Arguments:
            - test_loader: data loader for testing data. By the dataset of this server will be used.
        Returns:
            - A diction containing testing results.
        """
        raise NotImplementedError
```

### Group

```python
class ParameterServerGroup:
    r"""
    Overview:
        Base implementation of the group in federated learning.
    """

    def __init__(self, args: dict, logger: Logger):
        r"""
        Overview:
            Lazy initialization of group.
            To complete the initialization process, please call `self.initialization()` after server and all clients
        are initialized.
        Arguments:
            - args: arguments in dict type.
            - logger: logger for this group
        Returns:
            - None
        """
        self.clients = []
        self.server = None
        self.args = args
        self.logger = logger
        self._time = time.time()

    def initialize(self) -> None:
        r"""
        Overview:
            In this function, several things will be done:
            1) Set ``fed_key`` in each client is determined, determine which parameters should be included for federated
        learning.
            2) ``glob_dict`` in the server is determined, which is exactly a state dict with all keys in ``fed_keys``.
            3) Each client local model will be updated by ``glob_dict``.
        Returns:
            - None
        """
        # Step 1.
        fed_keys = get_parameters(
            self.clients[0].model, self.args.group.aggregation_parameters, return_dict=True
        ).keys()

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')
        glob_dict = {k: self.clients[0].model.state_dict()[k] for k in fed_keys}

        # Resume from the checkpoint if needed.
        if self.args.other.resume_path is not None:
            sd = dict(torch.load(self.args.other.resume_path))
            for k, v in sd.items():
                if k in glob_dict.keys():
                    glob_dict[k] = v
        self.server.glob_dict = glob_dict

        self.set_fed_keys()

        # Step 3.
        self.sync()

        # Logging model information.
        self.logger.logging(str(self.clients[0].model))
        self.logger.logging('All clients initialized.')
        self.logger.logging(
            'Parameter number in each model: {:.2f}M'.format(get_params_number(self.clients[0].model) / 1e6)
        )

    def append(self, client: ClientTemplate) -> None:
        r"""
        Overview:
            Append a client into the group.
        Arguments:
            - client: client to be added.
        Returns:
            - None
        """
        self.clients.append(client)

    def aggregate(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        r"""
        Overview:
            Aggregate all client models.
        Arguments:
            - train_round: current global epochs.
            - aggr_parameter_args: What parameters should be aggregated. If set to ``None``, the initialized setting \
                will be used.
        Returns:
            - trans_cost: uplink communication cost.
        """
        # Pick out the parameters for aggregation if needed.
        if aggr_parameter_args is not None:
            fed_keys_bak = self.clients[0].fed_keys
            new_fed_keys = get_parameters(self.clients[0].model, aggr_parameter_args, return_dict=True).keys()
            for client in self.clients:
                client.set_fed_keys(new_fed_keys)

        if self.args.group.aggregation_method == 'avg':
            trans_cost = fed_avg(self.clients, self.server)
            self.sync()
        else:
            raise KeyError('Unrecognized compression method: ' + self.args.group.aggregation_method)

        # Add logger for time per round.
        # This time is the interval between two times of executing this ``aggregate()`` function.
        time_per_round = time.time() - self._time
        self._time = time.time()
        self.logger.add_scalar('time/time_per_round', time_per_round, train_round)

        if aggr_parameter_args is not None:
            for client in self.clients:
                client.set_fed_keys(fed_keys_bak)

        return trans_cost

    def flush(self) -> None:
        r"""
        Overview:
            Reset this group and clear all server and clients.
        Returns:
            - None
        """
        self.clients = []
        self.server = None

    def sync(self) -> None:
        r"""
        Overview:
            Synchronize all local models, making their parameters same as global model.
        Returns:
            - None
        """
        state_dict = self.server.glob_dict
        for client in self.clients:
            client.update_model(state_dict)

    def set_fed_keys(self) -> None:
        r"""
        Overview:
            Set `fed_keys` of each client, determine which parameters should be included for federated learning
        Returns:
            - None
        """
        for client in self.clients:
            client.set_fed_keys(self.server.glob_dict.keys())
```

## Pipeline

A pipeline is the main entry of a specific algorithm. Currently, we have implemented two main pipelines:

- `generic_model_serial_pipeline` : This is the pipeline for generic federated learning.
- `personalized_model_serial_pipeline` : This is the pipeline for personalized federated learning.

Detailed implementation can be viewed at [here](https://github.com/FLAIR-Community/Fling/tree/main/fling/pipeline).