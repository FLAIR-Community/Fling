import os
import pickle
import random
import time
import warnings
from functools import reduce

import numpy as np
import torch
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


def calculate_mean_std(train_dataset, test_dataset):
    if train_dataset[0][0].shape[0] == 1:
        res = []
        res_std = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i][0]
            res.append(sample.mean())
            res_std.append(sample.std())

        for i in range(len(test_dataset)):
            sample = test_dataset[i][0]
            res.append(sample.mean())
            res_std.append(sample.std())

        return reduce(lambda x, y: x + y, res) / len(res), reduce(lambda x, y: x + y, res_std) / len(res)


class Logger:

    def __init__(self, path):
        self.path = os.path.join(path, 'txt_file.txt')
        if os.path.exists(path):
            warnings.warn(f'Logging directory exists. Current directory: {path}')
        if not os.path.exists(path):
            os.makedirs(path)
        with open(self.path, 'w') as f:
            f.write('Created at ' + time.asctime(time.localtime(time.time())) + '\n')

    def logging(self, s):
        print(s)
        with open(self.path, mode='a') as f:
            f.write('[' + time.asctime(time.localtime(time.time())) + ']    ' + s + '\n')


def save_config_file(config: dict, path: str) -> None:
    """
    Overview:
        save configuration to python file
    Arguments:
        - config (:obj:`dict`): Config dict
        - path (:obj:`str`): Path of target yaml
    """
    config_string = str(config)
    from yapf.yapflib.yapf_api import FormatCode
    config_string, _ = FormatCode(config_string)
    config_string = config_string.replace('inf,', 'float("inf"),')
    with open(path, "w") as f:
        f.write('exp_config = ' + config_string)

