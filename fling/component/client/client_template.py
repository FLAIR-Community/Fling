import copy
from typing import Iterable, Callable
from torch.optim.optimizer import Optimizer

from fling.model import get_model
from fling.utils import VariableMonitor


class ClientTemplate:
    r"""
    Overview:
        Template of client in Federated Learning.
    """

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable, test_dataset: Iterable = None) -> None:
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
        self.fed_keys = keys

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

    def preprocess_data(self, data: Iterable) -> dict:
        r"""
        Overview:
            Pre-process the data batch generated from dataset.
        Arguments:
            - data: raw data generated from dataset.
        Returns:
            - Data after pre-processing.
        """
        raise NotImplementedError

    def train(self, lr: float, device: str) -> dict:
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

    def finetune(self, lr: float, finetune_args: dict, device: str, finetune_eps: int) -> list:
        r"""
        Overview:
            The local fine-tuning process of a client.
        Arguments:
            - lr: learning rate of the training.
            - finetune_args: arguments for fine-tuning.
            - device: device for operating this function.
            - finetune_eps: epochs for finetuning.
        Returns:
            - A list of dictions containing fine-tuning results.
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
