import copy
import math
import pickle
import random
from functools import reduce
from typing import Union, Callable, Iterable, List, Dict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


def get_optimizer(weights: object, **kwargs) -> optim.Optimizer:
    # return the optimizer given optimizer config and model weights.
    name = kwargs.pop('name')
    optimizer_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    if not name.lower() in optimizer_dict.keys():
        raise ValueError(f'Unrecognized optimizer: {name}')
    return optimizer_dict[name.lower()](params=weights, **kwargs)


def get_params_number(net: nn.Module) -> int:
    # get total params number in nn.Module net
    res = 0
    for param in net.parameters():
        res += param.numel()
    return res


def save_file(obj: object, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_file(path: str) -> object:
    with open(path, 'rb') as f:
        return pickle.load(f)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def calculate_mean_std(train_dataset: Dataset, test_dataset: Dataset) -> tuple:
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


def get_weights(model: nn.Module,
                parameter_args: dict,
                return_dict: bool = False,
                include_non_param: bool = False) -> Union[List, Dict]:
    """
    Overview:
        Get model parameters, using the given ``parameter_args``.
    Arguments:
        - model: The model to extract parameters from.
        - parameter_args: The parameter argument that specify what parameters to extract.
        - return_dict: If ``True``, the returned type is diction; otherwise, the returned type is list.
        - include_non_param: If ``True``, all weights in ``model.state_dict()`` will be considered, including
            non-parameter weights (e.g. running_mean, running_var). Otherwise, only parameters of the model will
            be considered.
    """
    if parameter_args.name == 'all':
        use_keys = model.state_dict().keys()
    elif parameter_args.name == 'contain':
        keywords = parameter_args.keywords
        use_keys = []
        for kw in keywords:
            for k in model.state_dict():
                if kw in k:
                    use_keys.append(k)
        use_keys = list(set(use_keys))
    elif parameter_args.name == 'except':
        keywords = parameter_args.keywords
        use_keys = []
        for kw in keywords:
            for k in model.state_dict():
                if kw in k:
                    use_keys.append(k)
        use_keys = list(set(model.state_dict().keys()) - set(use_keys))
    else:
        raise ValueError(f'Unrecognized finetune parameter name: {parameter_args.name}')

    if include_non_param:
        if not return_dict:
            res = []
            for key in model.state_dict().keys():
                if key in use_keys:
                    res.append(model.state_dict()[key])
        else:
            res = {}
            for key in model.state_dict().keys():
                if key in use_keys:
                    res[key] = model.state_dict()[key]
    else:
        if not return_dict:
            res = []
            for key, param in model.named_parameters():
                if key in use_keys:
                    res.append(param)
        else:
            res = {}
            for key, param in model.named_parameters():
                if key in use_keys:
                    res[key] = param

    return res


def balanced_softmax_loss(
        labels: torch.LongTensor,
        logits: torch.FloatTensor,
        sample_per_class: torch.LongTensor,
        reduction: str = "mean"
) -> torch.Tensor:
    r"""
    Overview:
        Compute the Balanced Softmax Loss between ``logits`` and the ground truth ``labels``.
    Arguments:
        labels: ground truth labels. Shape: ``[B]``.
        logits: predicted logits. Shape: ``[B, C]``.
        sample_per_class: number of samples in this client. Shape: `[C]`.
        reduction: reduction method of loss, one of "none", "mean", "sum".
    Returns:
      loss: Calculated balanced softmax loss.
    """
    spc = sample_per_class.float()
    spc = spc.unsqueeze(0)
    logits = logits + torch.log(spc)
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class LRScheduler:

    def __init__(self, base_lr: float, args: dict):
        self.args = args
        self.lr = base_lr

    def get_lr(self, train_round: int) -> float:
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


def get_activation(name: str, **kwargs) -> Callable:
    func_dict = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}
    try:
        return func_dict[name](**kwargs)
    except KeyError:
        raise ValueError(f'Unrecognized activation function name: {name}')


def get_model_difference(
        model_a_param: Union[dict, Iterable, torch.Tensor],
        model_b_param: Union[dict, Iterable, torch.Tensor],
        norm_p: int = 2
) -> torch.Tensor:
    r"""
    Overview:
        Calculation the model difference of ``model_a_param`` and ``model_b_param``.
        The difference is calculated as the p-norm of ``model_a_param - model_b_param``
    Arguments:
        model_a_param: the parameters of model A.
        model_b_param: the parameters of model B.
        norm_p: the p-norm to calculate the difference.
    Returns:
        res: the calculated difference norm.
    """
    res = 0
    if isinstance(model_a_param, torch.Tensor) and isinstance(model_b_param, torch.Tensor):
        tmp_res = torch.norm(model_a_param - model_b_param, p=norm_p)
        if torch.isnan(tmp_res) or torch.isinf(tmp_res):
            raise ValueError('Nan or inf encountered in calculating norm.')
        res += tmp_res
    elif isinstance(model_a_param, dict) and isinstance(model_b_param, dict):
        for key, val in model_a_param.items():
            tmp_res = torch.norm(val - model_b_param[key], p=norm_p)
            # Dealing with special conditions.
            if torch.isnan(tmp_res) or torch.isinf(tmp_res):
                continue
            res += tmp_res
    elif isinstance(model_a_param, Iterable) and isinstance(model_b_param, Iterable):
        for para1, para2 in zip(model_a_param, model_b_param):
            tmp_res = torch.norm(para1 - para2, p=norm_p)
            # Dealing with special conditions.
            if torch.isnan(tmp_res) or torch.isinf(tmp_res):
                continue
            res += tmp_res
    else:
        raise TypeError(
            f'Unrecognized type for calculating model difference.'
            f' Model A: {type(model_a_param)}, model B: {type(model_b_param)}'
        )
    return res


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def model_add(model1: nn.Module, model2: nn.Module) -> nn.Module:
    ret = copy.deepcopy(model1)
    sd1, sd2 = model1.state_dict(), model2.state_dict()
    ret.load_state_dict({k: sd1[k] + sd2[k] for k in sd1.keys()})
    return ret


def model_sub(model1: nn.Module, model2: nn.Module) -> nn.Module:
    ret = copy.deepcopy(model1)
    sd1, sd2 = model1.state_dict(), model2.state_dict()
    ret.load_state_dict({k: sd1[k] - sd2[k] for k in sd1.keys()})
    return ret


def model_mul(scalar: float, model: nn.Module) -> nn.Module:
    ret = copy.deepcopy(model)
    sd = model.state_dict()
    ret.load_state_dict({k: scalar * sd[k] for k in sd.keys()})
    return ret
