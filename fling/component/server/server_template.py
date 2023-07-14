from typing import Iterable
from torch.utils.data import DataLoader
import torch.nn as nn


class ServerTemplate:
    r"""
    Overview:
        Template of server in Federated Learning.
    """

    def __init__(self, args: dict, test_dataset: Iterable):
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

    def apply_grad(self, grad: dict, lr: float = 1.) -> None:
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

    def test_step(self, model, batch_data, criterion, monitor) -> None:
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

    def test(self, model: nn.Module, test_loader: DataLoader = None) -> dict:
        r"""
        Overview:
            The local testing process of a client.
        Arguments:
            - test_loader: data loader for testing data. By the dataset of this server will be used.
        Returns:
            - A diction containing testing results.
        """
        raise NotImplementedError
