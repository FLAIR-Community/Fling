import copy
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedcac_client')
class FedCACClient(BaseClient):
    """
    Overview:
        This class is the base implementation of client in 'Bold but Cautious: Unlocking the Potential of Personalized
        Federated Learning through Cautiously Aggressive Collaboration' (FedCAC).
    """

    def __init__(self, args, client_id, train_dataset, test_dataset=None):
        """
        Initializing train dataset, test dataset(for personalized settings).
        """
        super(FedCACClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.critical_parameter = None  # record the critical parameter positions in FedCAC
        self.customized_model = copy.deepcopy(self.model)  # customized global model

    def train(self, lr, device=None, train_args=None):
        """
        Local training.
        """
        # record the model before local updating, used for critical parameter selection
        initial_model = copy.deepcopy(self.model)

        # local update for several local epochs
        mean_monitor_variables = super().train(lr, device, train_args)

        # select the critical parameters
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.learn.tau
        )

        return mean_monitor_variables

    def evaluate_critical_parameter(self, prevModel: nn.Module, model: nn.Module,
                                    tau: int) -> Tuple[torch.Tensor, list, list]:
        r"""
        Overview:
            Implement critical parameter selection.
        """
        global_mask = []  # mark non-critical parameter
        local_mask = []  # mark critical parameter
        critical_parameter = []

        self.model.to(self.device)
        prevModel.to(self.device)

        # select critical parameters in each layer
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            g = (param.data - prevparam.data)
            v = param.data
            c = torch.abs(g * v)

            metric = c.view(-1)
            num_params = metric.size(0)
            nz = int(tau * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            # if threshold equals 0, select minimal nonzero element as threshold
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-20]
                if len(new_metric) == 0:  # this means all items in metric are zero
                    print(f'Abnormal!!! metric:{metric}')
                else:
                    thresh = new_metric.sort()[0][0]

            # Get the local mask and global mask
            mask = (c >= thresh).int()
            global_mask.append((c < thresh).int())
            local_mask.append(mask)
            critical_parameter.append(mask.view(-1))
        model.zero_grad()
        critical_parameter = torch.cat(critical_parameter)

        self.model.to('cpu')
        prevModel.to('cpu')

        return critical_parameter, global_mask, local_mask
