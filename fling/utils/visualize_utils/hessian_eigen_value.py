from typing import Sequence, List, Dict
import copy

import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader


def _get_first_grad(loss: torch.Tensor, w: List) -> Sequence:
    """
    Calculate: g_i = \\frac{dL}{dW_i}
    """
    return grad(loss, w, retain_graph=True, create_graph=True)


def _get_hv(g: Sequence, w: Sequence, v: Sequence) -> Sequence:
    """
    Calculate: Hv = \\frac{d(gv)}{dW_i}
    """
    assert len(w) == len(v)
    return grad(g, w, grad_outputs=v, retain_graph=True)


def _normalize(vs: Sequence) -> None:
    """
    Normalize vectors in ``vs``.
    """
    for i in range(len(vs)):
        vs[i] = vs[i] / torch.norm(vs[i])


def _calc_loss_value(
    model: nn.Module, data_loader: DataLoader, device: str, criterion: nn.Module = nn.CrossEntropyLoss()
):
    # Given a model and corresponding dataset, calculate the mean loss value.
    model.eval()
    tot_loss = []
    for _, (data) in enumerate(data_loader):
        data_x, data_y = data['input'].to(device), data['class_id'].to(device)
        pred_y = model(data_x)
        loss = criterion(pred_y, data_y)
        tot_loss.append(loss)
    tot_loss = torch.stack(tot_loss, dim=0)
    return torch.mean(tot_loss)


def _rayleigh_quotient(hv: Sequence, v: Sequence) -> List:
    """
    Calculate: \\lambda = \\frac{v^THv}{v^Tv}
    """
    return [
        ((torch.flatten(v[i].T) @ torch.flatten(hv[i])) / (torch.flatten(v[i].T) @ torch.flatten(v[i]))).item()
        for i in range(len(hv))
    ]


def calculate_hessian_dominant_eigen_values(
        model: nn.Module, iter_num: int, dataloader: DataLoader, device: str
) -> Dict:
    """
    Overview:
        Using power iteration to calculate each dominant eigen value of each layer in the model.
        Reference paper: HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision
        <link https://arxiv.org/pdf/1905.03696.pdf link>
    Arguments:
        model: The neural network that calculates ``loss``.
        iter_num: Number of iterations using power iteration.
        dataloader: The dataloader used to calculate hessian eigen values.
        device: The device to run on, such as ``"cuda"`` or ``"cpu"``.
    Returns:
        A diction of dominant eigen values for each layer.
    """
    # Copy the original model.
    orig_model = model
    model = copy.deepcopy(model).to(device)

    # Calculate loss value using given data.
    loss = _calc_loss_value(model, data_loader=dataloader, device=device)

    # Calculate eigen values and return.
    # Flatten the parameter weights.
    ws = dict(model.named_parameters())
    keys = list(ws.keys())
    ws = list(ws.values())

    # Calculate grad.
    g = _get_first_grad(loss, ws)

    # Initialize vs and normalize them.
    vs = [torch.randn_like(g[i]) for i in range(len(g))]
    _normalize(vs)

    # Power iteration.
    for i in range(iter_num):
        hv = _get_hv(g, ws, vs)
        vs = [hv[i].detach() for i in range(len(hv))]
        _normalize(vs)

    # Calculate eigen values.
    hv = _get_hv(g, ws, vs)
    lambdas = _rayleigh_quotient(hv, vs)
    dict_lambdas = {keys[i]: lambdas[i] for i in range(len(lambdas))}

    return dict_lambdas
