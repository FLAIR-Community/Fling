import time
import copy
import torch

from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger, get_weights, get_params_number
from fling.component.group import ParameterServerGroup
from fling.component.server import ServerTemplate
from functools import reduce


@GROUP_REGISTRY.register('scaffold_group')
class SCAFFOLDServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in SCAFFOLD.
    """

    def __init__(self, args: dict, logger: Logger):
        super(SCAFFOLDServerGroup, self).__init__(args, logger)
        # To be consistent with the existing pipeline interface. group maintains an epoch counter itself.
        self.epoch = -1

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
        fed_keys = get_weights(
            self.clients[0].model, self.args.group.aggregation_parameters, return_dict=True, include_non_param=True
        ).keys()

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')

        glob_dict = {k: self.clients[0].model.state_dict()[k] for k in fed_keys}
        c = {k: torch.zeros_like(self.clients[0].model.state_dict()[k]) for k in fed_keys}

        # Resume from the checkpoint if needed.
        if self.args.other.resume_path is not None:
            sd = dict(torch.load(self.args.other.resume_path))
            for k, v in sd.items():
                if k in glob_dict.keys():
                    glob_dict[k] = v

        self.server.glob_dict = glob_dict
        self.server.c = c

        self.set_fed_keys()

        # Step 3.
        self.sync()

        # Logging model information.
        self.logger.logging(str(self.clients[0].model))
        self.logger.logging('All clients initialized.')
        self.logger.logging(
            'Parameter number in each model: {:.2f}M'.format(get_params_number(self.clients[0].model) / 1e6)
        )

    def sync(self) -> None:
        r"""
        Overview:
            Synchronize all local models, making their parameters same as global model.
        Returns:
            - None
        """
        state_dict = self.server.glob_dict
        c = self.server.c
        for client in self.clients:
            client.update_model(state_dict)
            client.update_c(c)

    def aggregate(
            self, train_round: int, participate_clients_ids: list = None, aggr_parameter_args: dict = None
    ) -> int:
        r"""
        Overview:
            Aggregate all participating client models.
        Arguments:
            - train_round: current global epochs.
            - participate_clients_ids: A index list record which clients are participating
            - aggr_parameter_args: What parameters should be aggregated. If set to ``None``, the initialized setting \
                will be used.
        Returns:
            - trans_cost: uplink communication cost.
        """
        if participate_clients_ids is None:
            participate_clients_ids = list(range(self.args.client.client_num))
        participate_clients = [self.clients[i] for i in participate_clients_ids]
        # Pick out the parameters for aggregation if needed.
        if aggr_parameter_args is not None:
            fed_keys_bak = self.clients[0].fed_keys
            new_fed_keys = get_weights(
                self.clients[0].model, aggr_parameter_args, return_dict=True, include_non_param=True
            ).keys()
            for client in participate_clients:
                client.set_fed_keys(new_fed_keys)

        if self.args.group.aggregation_method == 'avg':
            K = len(participate_clients)
            N = len(self.clients)
            keys = []
            bn_keys = []

            for k, v in self.clients[0].model.named_parameters():
                keys.append(k)
            for k in self.clients[0].model.state_dict():
                if k not in keys:
                    bn_keys.append(k)

            # Aggregate c and y
            avg_delta_y = {
                k: reduce(lambda x, y: x + y, [client.delta_y[k] / K for client in participate_clients])
                for k in keys
            }
            avg_delta_c = {
                k: reduce(lambda x, y: x + y, [client.delta_c[k] / K for client in participate_clients])
                for k in keys
            }
            avg_bn_val = {
                k: reduce(lambda x, y: x + y, [client.model.state_dict()[k] / K for client in participate_clients])
                for k in bn_keys
            }
            trans_cost = 4 * sum(
                N * (self.clients[0].delta_y[k].numel() + self.clients[0].delta_c[k].numel()) for k in keys
            )

            for k in keys:
                self.server.glob_dict[k] += self.args.learn.server_lr * avg_delta_y[k]
                self.server.c[k] += K / N * avg_delta_c[k]
            for k in bn_keys:
                self.server.glob_dict[k] = avg_bn_val[k]

            self.sync()
        else:
            raise KeyError('Unrecognized compression method: ' + self.args.group.aggregation_method)

        # Add logger for time per round.
        # This time is the interval between two times of executing this ``aggregate()`` function.
        time_per_round = time.time() - self._time
        self._time = time.time()
        self.logger.add_scalar('time/time_per_round', time_per_round, train_round)

        if aggr_parameter_args is not None:
            for client in participate_clients:
                client.set_fed_keys(fed_keys_bak)

        return trans_cost
