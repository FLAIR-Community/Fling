import math
import pickle
import random
from functools import reduce

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def get_optimizer(name, lr, momentum, weights):
    if name.lower() == 'sgd':
        return optim.SGD(params=weights, momentum=momentum, lr=lr)
    elif name.lower() == 'adam':
        return optim.Adam(params=weights, lr=lr)
    else:
        print('Unrecognized optimizer: ' + name)
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


def get_finetune_parameters(model, finetune_args):
    if finetune_args.name == 'all':
        use_keys = model.state_dict().keys()
    elif finetune_args.name == 'contain':
        keywords = finetune_args.keywords
        use_keys = []
        for kw in keywords:
            for k in model.state_dict():
                if kw in k:
                    use_keys.append(k)
        use_keys = list(set(use_keys))
    elif finetune_args.name == 'except':
        keywords = finetune_args.keywords
        use_keys = []
        for kw in keywords:
            for k in model.state_dict():
                if kw in k:
                    use_keys.append(k)
        use_keys = list(set(model.state_dict().keys()) - set(use_keys))
    else:
        raise ValueError(f'Unrecognized finetune parameter name: {finetune_args.name}')

    res = []
    for key, param in model.named_parameters():
        if key in use_keys:
            res.append(param)
    return res


class LRScheduler:

    def __init__(self, args):
        self.args = args.learn.scheduler
        self.lr = args.learn.optimizer.lr

    def get_lr(self, train_round):
        if self.args.name == 'fix':
            return self.lr
        elif self.args.name == 'linear':
            return self.lr - self.args.decay_coefficient * train_round
        elif self.args.name == 'exp':
            return self.lr * (self.args.decay_coefficient ** train_round)
        elif self.args.name == 'cos':
            min_lr = self.args.min_lr
            if train_round > self.args.decay_round:
                return min_lr
            decay_ratio = train_round / self.args.decay_round
            assert 0 <= decay_ratio <= 1
            coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coefficient * (self.lr - min_lr)
        else:
            raise ValueError(f'Unrecognized lr scheduler: {self.args.name}')


def get_activation(name, **kwargs):
    func_dict = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}
    try:
        return func_dict[name](**kwargs)
    except KeyError:
        raise ValueError(f'Unrecognized activation function name: {name}')
