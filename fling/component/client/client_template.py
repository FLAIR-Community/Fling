import copy

from fling.model import get_model


class ClientTemplate:
    """
    Overview:
    This class is the base implementation of client in Federated Learning.
    Typically, a client need to have these functions.
    ``train``: A client need to define the local training process.
    ``test``: A client need to define how to test the local model given a dataset.
    ``finetune``: A client need to define how to finetune the local model (usually used in Personalized Federated Learning)
    If users want to define a new client class, it is recommended to inherit this class.
    """

    def __init__(self, args, train_dataset, client_id):
        """
        Initializing train dataset, test dataset(for personalized settings), constructing model and other configurations.
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

    def set_fed_keys(self, keys):
        """
        Set the attribute ``self.fed_keys``.
        Only weights in ``self.fed_keys`` will be collaboratively trained using Federated Learning.
        """
        self.fed_keys = keys

    def update_model(self, dic):
        """
        Using the ``dic`` to update the state_dict of the local model of this client.
        For keys not existed in ``dic``, the value will be retained.
        """
        dic = copy.deepcopy(dic)
        state_dict = self.model.state_dict()
        state_dict.update(dic)

        self.model.load_state_dict(state_dict)

    def get_state_dict(self, keys):
        """
        Get the state dict of local model.
        """
        state_dict = self.model.state_dict()
        return {k: state_dict[k] for k in keys}

    def train_step(self, batch_data, criterion, monitor, optimizer):
        raise NotImplementedError

    def test_step(self, batch_data, criterion, monitor):
        raise NotImplementedError

    def preprocess_data(self, data):
        raise NotImplementedError

    def train(self, lr):
        raise NotImplementedError

    def finetune(self, lr, finetune_args, finetune_eps=None):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
