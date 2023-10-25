import time
import copy
import torch

from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger
from fling.component.group import ParameterServerGroup
from fling.utils.utils import weight_flatten

@GROUP_REGISTRY.register('fedamp_group')
class FedAMPServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in FedAMP.
    """

    def __init__(self, args: dict, logger: Logger):
        super(FedAMPServerGroup, self).__init__(args, logger)
        # FedAMP auguments
        self.alphaK = args.learn.alphaK
        self.sigma = args.learn.sigma
        self.lamda = args.learn.lamda
        self.client_ws = [None for i in range(self.args.client.client_num)]    # maintain all clients' personalized models
        self.client_us = [None for i in range(self.args.client.client_num)]    # aggregated model for each client

    def sync(self) -> None:
        r"""
        Overview:
            Send customized global models to each client
        """
        if self.client_us[0] is None:
            super().sync()  # Called during system initialization

        else:
            for idx, client in enumerate(self.clients):
                client.client_u = copy.deepcopy(self.client_us[idx])

    def initialize(self) -> None:
        super().initialize()
        self.client_ws = [copy.deepcopy(self.clients[i].model) for i in range(self.args.client.client_num)]
        self.client_us = [copy.deepcopy(self.clients[i].model) for i in range(self.args.client.client_num)]


    def receive_models(self):
        r"""
        Overview:
            Receive personalized models from each client
        """
        for idx, client in enumerate(self.clients):
            self.client_ws[idx] = copy.deepcopy(client.model)

    def aggregate(self, train_round: int) -> int:
        r"""
        Overview:
            Aggregate customized global models and send to each client
        """
        # recieve models from clients
        self.receive_models()
        for i in range(self.args.client.client_num):
            self.client_ws[i].to(self.args.learn.device)
            self.client_us[i].to(self.args.learn.device)

        # aggregate models
        weights = [weight_flatten(mw) for mw in self.client_ws]
        for i, mu in enumerate(self.client_us):  # calculate u for each client
            for param in mu.parameters():  # set zero for each parameter
                param.data = torch.zeros_like(param.data)

            coef = torch.zeros(self.args.client.client_num)
            for j, mw in enumerate(self.client_ws):
                if i == j: continue
                sub = weights[i] - weights[j]
                sub = torch.dot(sub, sub)
                coef[j] = self.args.learn.alphaK * self.e(sub)
            coef[i] = 1 - torch.sum(coef)

            for j, mw in enumerate(self.client_ws):
                for param, param_j in zip(mu.parameters(), mw.parameters()):
                    param.data += coef[j] * param_j

        for i in range(self.args.client.client_num):
            self.client_ws[i].to('cpu')
            self.client_us[i].to('cpu')

        # send to all clients
        self.sync()

        # perform alphaK decay
        if train_round % self.args.learn.decay_frequency == 0 and train_round != 0:
            self.args.learn.alphaK *= self.args.learn.decay_rate

        # calculate communication cost
        trans_cost = 0
        state_dict = self.clients[0].model.state_dict()
        for k in self.clients[0].fed_keys:
            trans_cost += self.args.client.client_num * state_dict[k].numel()
        # 1B = 32bit
        return 4 * trans_cost
    def e(self, x):
        r"""
        Overview:
            The derivative of attention-inducing function in FedAMP
        """
        return torch.exp(-x/self.sigma)/self.sigma
