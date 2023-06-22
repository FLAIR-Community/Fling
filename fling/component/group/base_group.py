from fling.utils import get_params_number
from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY


@GROUP_REGISTRY.register('base_group')
class ParameterServerGroup:
    """
    A container to hold clients.
    """

    def __init__(self, args, logger):
        self.clients = []
        self.server = None
        self.method = None
        self.args = args
        self.logger = logger

    def initialize(self):
        """
        Overview:
        In this function, several things will be done:
        1) ``fed_key`` in each client is determined.
        2) ``glob_dict`` in the server is determined, which is exactly a state dict with all keys in ``fed_keys``.
        3) Each client local model will be updated by ``glob_dict``.
        """
        # Step 1.
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

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')
        glob_dict = {k: self.clients[0].model.state_dict()[k] for k in fed_keys}
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

    def append(self, item):
        """
        append: add a client into the container
        :param item: the client to be added
        :return: None
        """
        self.clients.append(item)

    def aggregate(self, train_round, tb_logger):
        """
        aggregate: applying an aggregation method to update the global model
        :return: None
        """
        if self.args.group.aggregation_method == 'avg':
            trans_cost = fed_avg(self.clients, self.server)
            self.sync()
        else:
            print('Unrecognized compression method: ' + self.method)
            assert False
        return trans_cost

    def flush(self):
        """
        flush all the clients
        :return: None
        """
        self.clients = []

    def sync(self):
        """
        given a global state_dict, require all clients' model is set equal as server
        :return: None
        """
        state_dict = self.server.glob_dict
        for client in self.clients:
            client.update_model(state_dict)

    def set_fed_keys(self):
        for client in self.clients:
            client.set_fed_keys(self.server.glob_dict.keys())
