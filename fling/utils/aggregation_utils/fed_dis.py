import copy

import torch
from torch import nn, optim

from fling.component.server import ServerTemplate
from fling.utils.aggregation_utils import tensor_reduce


def linear_regression(in_channel, out_channel, with_bias, activation, weights, biases, device):
    target_models = []
    if with_bias:
        assert len(weights) == len(biases)
    global_model = nn.Linear(in_channel, out_channel, bias=with_bias)
    global_model.weight.data = torch.zeros_like(global_model.weight)
    if with_bias:
        global_model.bias.data = torch.zeros_like(global_model.bias)

    for i in range(len(weights)):
        new_model = nn.Linear(in_channel, out_channel, bias=with_bias)
        new_model.weight.data = weights[i].clone()
        if with_bias:
            new_model.bias.data = biases[i].clone()
        target_models.append(new_model)
        global_model.weight.data += new_model.weight
        if with_bias:
            global_model.bias.data += new_model.bias

    global_model.weight.data /= len(weights)
    if with_bias:
        global_model.bias.data /= len(weights)

    global_model.to(device)
    for i in range(len(target_models)):
        target_models[i] = target_models[i].to(device)

    optimizer = optim.LBFGS(global_model.parameters(), lr=0.1, max_iter=20, history_size=10)

    if activation == 'relu':
        activation_function = torch.relu
    elif activation == 'sigmoid':
        activation_function = torch.sigmoid
    elif activation == 'tanh':
        activation_function = torch.tanh
    else:
        assert False

    def closure():
        optimizer.zero_grad()
        x = torch.randn(8192, in_channel).to(device)
        global_output = activation_function(global_model(x))

        loss = 0
        for model in target_models:
            local_output = activation_function(model(x))
            loss += torch.mean((global_output - local_output) ** 2)

        loss.backward()
        return loss

    best_ckpt = [1e9, None]
    for epoch in range(1000):
        loss = optimizer.step(closure)
        if loss.item() <= best_ckpt[0]:
            best_ckpt[0] = loss.item()
            best_ckpt[1] = copy.deepcopy(global_model).to('cpu')

    if with_bias:
        return best_ckpt[1].weight.data, best_ckpt[1].bias.data
    else:
        return best_ckpt[1].weight.data


def conv_regression(in_channel, out_channel, with_bias, activation, weights, biases, device):
    pass


def fed_dis(clients: list, server: ServerTemplate) -> int:
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

    parameter_keys = clients[0].fed_keys
    waiting_keys = []
    state_dicts = [clients[i].model.state_dict() for i in range(len(clients))]
    final_global_dict = {}

    while len(parameter_keys) > 0:
        target_key = parameter_keys.pop(0)
        param_shape = state_dicts[0][target_key].shape
        if len(param_shape) == 2:  # Linear layer.
            weights = [state_dicts[i][target_key] for i in range(len(clients))]
            bias_key_name = '.'.join(target_key.split('.')[:-1]) + '.bias'
            if bias_key_name in parameter_keys:
                parameter_keys.remove(bias_key_name)
                biases = [state_dicts[i][bias_key_name] for i in range(len(clients))]
            elif bias_key_name in waiting_keys:
                waiting_keys.remove(bias_key_name)
                biases = [state_dicts[i][bias_key_name] for i in range(len(clients))]
            else:
                biases = None
            if biases is not None:
                global_w, global_b = linear_regression(param_shape[1], param_shape[0], with_bias=biases is not None,
                                                       weights=weights, biases=biases,
                                                       device=clients[0].args.learn.device,
                                                       activation=clients[0].args.model.activation)
                final_global_dict[target_key] = global_w
                final_global_dict[bias_key_name] = global_b
            else:
                print(target_key, bias_key_name, parameter_keys)
                global_w = linear_regression(param_shape[1], param_shape[0], with_bias=biases is not None,
                                             weights=weights, biases=biases,
                                             device=clients[0].args.learn.device,
                                             activation=clients[0].args.model.activation)
                final_global_dict[target_key] = global_w

        elif len(param_shape) == 4:  # Conv layer.
            # TODO
            weights = [state_dicts[i][target_key] for i in range(len(clients))]
            bias_key_name = '.'.join(target_key.split('.')[:-1])
            if bias_key_name in parameter_keys:
                parameter_keys.remove(bias_key_name)
                biases = [state_dicts[i][bias_key_name] for i in range(len(clients))]
            elif bias_key_name in waiting_keys:
                waiting_keys.remove(bias_key_name)
                biases = [state_dicts[i][bias_key_name] for i in range(len(clients))]
            else:
                biases = None
            if biases is not None:
                global_w, global_b = linear_regression(param_shape[1], param_shape[0], with_bias=biases is not None,
                                                       weights=weights, biases=biases,
                                                       device=clients[0].args.learn.device,
                                                       activation=clients[0].args.model.activation)
                final_global_dict[target_key] = global_w
                final_global_dict[bias_key_name] = global_b
            else:
                global_w = linear_regression(param_shape[1], param_shape[0], with_bias=biases is not None,
                                             weights=weights, biases=biases,
                                             device=clients[0].args.learn.device,
                                             activation=clients[0].args.model.activation)
                final_global_dict[target_key] = global_w
        else:
            waiting_keys.append(target_key)

    for k in waiting_keys:
        final_global_dict[k] = tensor_reduce(
                lambda x, y: x + y,
                [client.model.state_dict()[k] for client in clients],
                device=clients[0].args.learn.device
            )

    # Calculate the ``trans_cost``.
    trans_cost = 0
    state_dict = clients[0].model.state_dict()
    for k in clients[0].fed_keys:
        trans_cost += len(clients) * state_dict[k].numel()
    # 1B = 32bit
    return 4 * trans_cost
