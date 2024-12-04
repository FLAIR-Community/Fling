from .utils import tensor_reduce
from fling.component.server import ServerTemplate


def fed_avg(clients: list, server: ServerTemplate) -> int:
    r"""
    Overview:
        Use the average method to aggregate parameters in different client models.
        Note that only the keys in ``server.glob_dict`` will be aggregated.
        Parameters besides these keys will be retained in each client.
    Arguments:
        clients: a list of clients that is needed to be aggregated in this round.
        server: The parameter server of these clients.
    Returns:
        trans_cost: the total uplink cost in this communication round.
    """

    # The ``sample_num`` refers to the number of data in each client.
    # FedAvg will use a weighted-averaging algorithm to average client models according to their ``sample_num``
    total_samples = sum([client.sample_num for client in clients])
    # Weighted-averaging.
    server.glob_dict = {
        k: tensor_reduce(
            lambda x, y: x + y,
            [client.sample_num / total_samples * client.model.state_dict()[k] for client in clients],
            device=clients[0].args.learn.device
        )
        for k in clients[0].fed_keys
    }
    # Calculate the ``trans_cost``.
    trans_cost = 0
    state_dict = clients[0].model.state_dict()
    for k in clients[0].fed_keys:
        trans_cost += len(clients) * state_dict[k].numel()
    # 1B = 32bit
    return 4 * trans_cost


def cross_domain_fed_avg(clients: dict, domains: dict, num_user: int, server: ServerTemplate) -> int:
    r"""
    Overview:
        The average aggregation method used for cross-domain federated learning scenarios.
        Use the average method to aggregate parameters in different client models.
        Note that only the keys in ``server.glob_dict`` will be aggregated.
        Parameters besides these keys will be retained in each client.
    Arguments:
        clients: a dict of clients that is needed to be aggregated in this round.
        domains: Used to indicate the domain origin of the clients.
        num_user: Used to index among clients within the same domain.
        server: The parameter server of these clients.
    Returns:
        trans_cost: the total uplink cost in this communication round.
    """

    # The ``sample_num`` refers to the number of data in each client.
    # FedAvg will use a weighted-averaging algorithm to average client models according to their ``sample_num``
    total_samples = sum(clients[domain][client].sample_num for domain in domains for client in range(num_user))
    # Weighted-averaging.
    server.glob_dict = {
        k: tensor_reduce(
            lambda x, y: x + y, [
                clients[domain][client].sample_num / total_samples * clients[domain][client].model.state_dict()[k]
                for domain in domains for client in range(num_user)
            ],
            device=clients[domains[0]][0].args.learn.device
        )
        for k in clients[domains[0]][0].fed_keys
    }

    # Calculate the ``trans_cost``.
    trans_cost = 0
    state_dict = clients[domains[0]][0].model.state_dict()
    for k in clients[domains[0]][0].fed_keys:
        trans_cost += len(domains) * num_user * state_dict[k].numel()
    # 1B = 32bit
    return 4 * trans_cost
