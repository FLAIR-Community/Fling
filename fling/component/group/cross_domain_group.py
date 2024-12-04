import time
import torch

from fling.utils import get_params_number
from fling.utils.compress_utils import cross_domain_fed_avg
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger, get_weights
from fling.component.client import ClientTemplate
from fling.component.group import ParameterServerGroup


@GROUP_REGISTRY.register('cross_domain_group')
class CrossDomainServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in Cross-Domain Scenario.
    """

    def __init__(self, args: dict, logger: Logger):
        super(CrossDomainServerGroup, self).__init__(args, logger)
        self.domains = args.data.domains.split(',')
        self.num_user = args.client.client_num
        self.clients = {domain: [] for domain in self.domains}

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
            self.clients[self.domains[0]][0].model,
            self.args.group.aggregation_parameters,
            return_dict=True,
            include_non_param=True
        ).keys()

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')
        glob_dict = {k: self.clients[self.domains[0]][0].model.state_dict()[k] for k in fed_keys}

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
        self.logger.logging(str(self.clients[self.domains[0]][0].model))
        self.logger.logging('All clients initialized.')
        self.logger.logging(
            'Parameter number in each model: {:.2f}M'.format(
                get_params_number(self.clients[self.domains[0]][0].model) / 1e6
            )
        )

    def append(self, domain: str, client: ClientTemplate) -> None:
        r"""
        Overview:
            Append a client into the group.
        Arguments:
            - domain: the domain to which the new client belongs.
            - client: client to be added.
        Returns:
            - None
        """
        self.clients[domain].append(client)

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
            fed_keys_bak = self.clients[self.domains[0]][0].fed_keys
            new_fed_keys = get_weights(
                self.clients[self.domains[0]][0].model, aggr_parameter_args, return_dict=True, include_non_param=True
            ).keys()
            for domain in self.domains:
                for client in range(self.num_user):
                    self.clients[domain][client].set_fed_keys(new_fed_keys)

        if self.args.group.aggregation_method == 'avg':
            trans_cost = cross_domain_fed_avg(self.clients, self.domains, self.num_user, self.server)
            self.sync()
        else:
            raise KeyError('Unrecognized compression method: ' + self.args.group.aggregation_method)

        # Add logger for time per round.
        # This time is the interval between two times of executing this ``aggregate()`` function.
        time_per_round = time.time() - self._time
        self._time = time.time()
        self.logger.add_scalar('time/time_per_round', time_per_round, train_round)

        if aggr_parameter_args is not None:
            for domain in self.domains:
                for client in range(self.num_user):
                    self.clients[domain][client].set_fed_keys(fed_keys_bak)

        return trans_cost

    def sync(self) -> None:
        r"""
        Overview:
            Synchronize all local models, making their parameters same as global model.
        Returns:
            - None
        """
        state_dict = self.server.glob_dict
        for domain in self.domains:
            for client in range(self.num_user):
                self.clients[domain][client].update_model(state_dict)

    def set_fed_keys(self) -> None:
        r"""
        Overview:
            Set `fed_keys` of each client, determine which parameters should be included for federated learning
        Returns:
            - None
        """
        for domain in self.domains:
            for client in range(self.num_user):
                self.clients[domain][client].set_fed_keys(self.server.glob_dict.keys())
