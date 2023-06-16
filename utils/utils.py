import pickle
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim


def get_optimizer(name, lr, momentum, weights):
    if name.lower() == 'sgd':
        return optim.SGD(params=weights, momentum=momentum, lr=lr)
    elif name.lower() == 'adam':
        return optim.Adam(params=weights, lr=lr)
    else:
        print('Unrecognized optimizer: ' + name)
        assert False


def get_loss(name):
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSE':
        return nn.MSELoss()
    elif name == 'gpt':
        return lambda x, y: x[1]
    else:
        print('Unrecognized loss: ' + name)
        assert False


def get_params_number(net):
    # get total params number in nn.Module net
    res = 0
    for param in net.parameters():
        res += param.numel()
    return res


def save_file(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
