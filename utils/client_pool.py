from utils.compressor import *


class ClientPool:
    """
    A container to hold clients.
    """

    def __init__(self, args):
        self.clients = []
        self.server = None
        self.method = None
        self.args = args

        self.last_grads = None

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
        if self.args.aggr_method == 'avg':
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

    def __getitem__(self, item):
        return self.clients[item]

    def sync(self):
        """
        given a global state_dict, require all clients' model is set equal as server
        :return: None
        """
        state_dict = self.server['glob_dict']
        for client in self.clients:
            client.update_model(state_dict)

    def set_up_fed_keys(self, keys):
        for client in self.clients:
            client.set_fed_keys(keys)

    def setup_compression_settings(self, method='avg', compress_ratio=40):
        self.method = method
        if method == 'powersgd':
            for client in self.clients:
                client.setup_powersgd(compress_ratio)

    def set_fed_keys(self):
        for client in self.clients:
            client.set_fed_keys(self.server['glob_dict'].keys())
