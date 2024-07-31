import time
import copy
import torch

from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger
from fling.component.group import ParameterServerGroup


@GROUP_REGISTRY.register('fedcac_group')
class FedCACServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in FedCAC.
    """

    def __init__(self, args: dict, logger: Logger):
        super(FedCACServerGroup, self).__init__(args, logger)
        # To be consistent with the existing pipeline interface. group maintains an epoch counter itself.
        self.epoch = -1

    def sync(self) -> None:
        r"""
        Overview:
            Perform the critical and non-critical parameter initialization steps in FedCAC.
        """
        if self.epoch == -1:
            super().sync()  # Called during system initialization
        else:
            tempGlobalModel = copy.deepcopy(self.clients[0].model)
            tempGlobalModel.load_state_dict(self.server.glob_dict)
            tempGlobalModel.to(self.args.learn.device)

            for client in range(self.args.client.client_num):
                index = 0
                self.clients[client].model.to(self.args.learn.device)
                self.clients[client].customized_model.to(self.args.learn.device)
                for (name1, param1), (name2, param2), (name3, param3) in zip(
                        self.clients[client].model.named_parameters(), tempGlobalModel.named_parameters(),
                        self.clients[client].customized_model.named_parameters()):
                    param1.data = self.clients[client].local_mask[index].to(self.args.learn.device).float() * param3.data + \
                                  self.clients[client].global_mask[index].to(self.args.learn.device).float() * param2.data
                    index += 1
                self.clients[client].model.to('cpu')
                self.clients[client].customized_model.to('cpu')
            tempGlobalModel.to('cpu')
        self.epoch += 1

    def get_customized_global_models(self) -> int:
        r"""
        Overview:
            Aggregating customized global models for clients to collaborate critical parameters.
        """
        assert type(self.args.learn.beta) == int and self.args.learn.beta >= 1
        overlap_buffer = [[] for i in range(self.args.client.client_num)]

        # calculate overlap rate between client i and client j
        for i in range(self.args.client.client_num):
            for j in range(self.args.client.client_num):
                if i == j:
                    continue
                overlap_rate = 1 - torch.sum(
                    torch.abs(
                        self.clients[i].critical_parameter.to(self.args.learn.device) -
                        self.clients[j].critical_parameter.to(self.args.learn.device)
                    )
                ) / float(torch.sum(self.clients[i].critical_parameter.to(self.args.learn.device)).cpu() * 2)
                overlap_buffer[i].append(overlap_rate)

        # calculate the global threshold
        overlap_buffer_tensor = torch.tensor(overlap_buffer)
        overlap_sum = overlap_buffer_tensor.sum()
        overlap_avg = overlap_sum / ((self.args.client.client_num - 1) * self.args.client.client_num)
        overlap_max = overlap_buffer_tensor.max()
        threshold = overlap_avg + (self.epoch + 1) / self.args.learn.beta * (overlap_max - overlap_avg)

        # calculate the customized global model for each client
        for i in range(self.args.client.client_num):
            w_customized_global = copy.deepcopy(self.clients[i].model.state_dict())
            collaboration_clients = [i]
            # find clients whose critical parameter locations are similar to client i
            index = 0
            for j in range(self.args.client.client_num):
                if i == j:
                    continue
                if overlap_buffer[i][index] >= threshold:
                    collaboration_clients.append(j)
                index += 1

            for key in w_customized_global.keys():
                for client in collaboration_clients:
                    if client == i:
                        continue
                    w_customized_global[key] += self.clients[client].model.state_dict()[key]
                w_customized_global[key] = torch.div(w_customized_global[key], float(len(collaboration_clients)))
            self.clients[i].customized_model.load_state_dict(w_customized_global)

        # Calculate the ``trans_cost``.
        trans_cost = 0
        state_dict = self.clients[0].model.state_dict()
        for k in state_dict.keys():
            trans_cost += self.args.client.client_num * state_dict[k].numel()
        return trans_cost

    def aggregate(self, train_round: int, participate_clients_ids: list = None) -> int:
        r"""
        Overview:
            Aggregate all client models.
            Generate customized global model for each client.
        Arguments:
            - train_round: current global epochs.
        Returns:
            - trans_cost: uplink communication cost.
        """
        if participate_clients_ids is None:
            participate_clients_ids = list(range(self.args.client.client_num))
        if self.args.group.aggregation_method == 'avg':
            trans_cost = fed_avg(self.clients, self.server)
            trans_cost += self.get_customized_global_models()
            self.sync()
        else:
            print('Unrecognized compression method: ' + self.args.group.aggregation_method)
            assert False

        # Add logger for time per round.
        # This time is the interval between two times of executing this ``aggregate()`` function.
        time_per_round = time.time() - self._time
        self._time = time.time()
        self.logger.add_scalar('time/time_per_round', time_per_round, train_round)

        return trans_cost
