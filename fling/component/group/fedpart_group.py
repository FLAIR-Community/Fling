import time
import torch

from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger, get_weights, get_params_number
from fling.component.group import ParameterServerGroup


@GROUP_REGISTRY.register('fedpart_group')
class FedPartServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in FedPart.
    """

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
            self.clients[0].model, self.args.group.aggregation_parameters, return_dict=True,
            include_non_param=False, prefix_mode=True
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
                self.clients[0].model, aggr_parameter_args, return_dict=True,
                include_non_param=False, prefix_mode=True
            ).keys()
            for client in participate_clients:
                client.set_fed_keys(new_fed_keys)

        if self.args.group.aggregation_method == 'avg':
            trans_cost = fed_avg(participate_clients, self.server)
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
