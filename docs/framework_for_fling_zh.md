# Fling 整体框架

## Fling 组件

### 概述

​	在 Fling 的设计中，我们提取了联邦学习过程中的三个主要组件，分别是：客户端（Client）、服务器（Server）和群组（Group）。

- **客户端（Client）:** 客户端指一个计算节点。每个客户端都有自己的私有数据集，并使用它来更新联邦学习模型。
- **服务器（Server）:** 服务器指的是中央计算节点。服务器通常用于存储全局模型、测试全局模型的性能。
- **群组（Group）:** 群组指的是由若干客户端和服务器（通常为一个服务器和多个客户端）组成的集合。对于同一群组中的客户端，它们将进行联邦学习。

下面是对这三个组件的详细介绍：

### 客户端（Client）

```python
<<<<<<< HEAD
=======
import copy
from typing import Callable, Iterable
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset

from fling.model import get_model
from fling.utils import VariableMonitor


>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e
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

<<<<<<< HEAD
    def train(self, lr: float, device: str) -> dict:
=======
    def train(self, lr: float, device: str, train_args: dict = None) -> dict:
>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e
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

### 服务器（Server）

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

### 群组（Group）

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
<<<<<<< HEAD
            1) Set ``fed_key`` in each client is determined, determine which parameters shoul be included for federated
=======
            1) Set ``fed_key`` in each client is determined, determine which parameters should be included for federated
>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e
        learning.
            2) ``glob_dict`` in the server is determined, which is exactly a state dict with all keys in ``fed_keys``.
            3) Each client local model will be updated by ``glob_dict``.
        Returns:
            - None
        """
        # Step 1.
<<<<<<< HEAD
        if self.args.group.aggregation_parameters.name == 'all':
            fed_keys = self.clients[0].model.state_dict().keys()
        elif self.args.group.aggregation_parameters.name == 'contain':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(fed_keys))
        elif self.args.group.aggregation_parameters.name == 'except':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(self.clients[0].model.state_dict().keys()) - set(fed_keys))
        else:
            raise ValueError(f'Unrecognized aggregation_parameters.name: {self.args.group.aggregation_parameters.name}')
=======
        fed_keys = get_parameters(
            self.clients[0].model, self.args.group.aggregation_parameters, return_dict=True
        ).keys()
>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e

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

<<<<<<< HEAD
    def aggregate(self, train_round: int) -> int:
=======
    def aggregate(self, train_round: int, aggr_parameter_args: dict = None) -> int:
>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e
        r"""
        Overview:
            Aggregate all client models.
        Arguments:
            - train_round: current global epochs.
<<<<<<< HEAD
        Returns:
            - trans_cost: uplink communication cost.
        """
=======
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

>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e
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

<<<<<<< HEAD
=======
        if aggr_parameter_args is not None:
            for client in self.clients:
                client.set_fed_keys(fed_keys_bak)

>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e
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
<<<<<<< HEAD
            Set `fed_keys` of each client, deterimine which parameters should be included for federated learning
=======
            Set `fed_keys` of each client, determine which parameters should be included for federated learning
>>>>>>> b9233a6a88a5c9c0d4a7bdd4a1bad951e858be2e
        Returns:
            - None
        """
        for client in self.clients:
            client.set_fed_keys(self.server.glob_dict.keys())
```

## Pipeline（流水线）

pipeline 是特定算法的主要入口。目前，我们已经实现了两个主要的 pipeline：

- `generic_model_serial_pipeline`：这是用于通用联邦学习的 pipeline。
- `personalized_model_serial_pipeline`：这是用于个性化联邦学习的 pipeline。

具体的实现可以点击[此处](https://github.com/kxzxvbk/Fling/tree/main/fling/pipeline)查看。
