import copy
from typing import Tuple, Dict
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


def _gen_rand_like(tensor: torch.Tensor) -> torch.Tensor:
    # Return a tensor whose shape is identical to the input tensor.
    # The returned tensor is a filled with Gaussian noise and the norm in each line is the same as the input.
    tmp = torch.rand_like(tensor)
    tmp = tmp / torch.norm(tmp, dim=1, keepdim=True)
    tmp = tmp * torch.norm(tensor, dim=1, keepdim=True)
    return tmp


def _calc_loss_value(
    model: nn.Module, data_loader: DataLoader, device: str, criterion: nn.Module = nn.CrossEntropyLoss()
):
    # Given a model and corresponding dataset, calculate the mean loss value.
    model = model.to(device)
    model.eval()
    tot_loss = []
    for _, (data_x, data_y) in enumerate(data_loader):
        data_x, data_y = data_x.to(device), data_y.to(device)
        pred_y = model(data_x)
        loss = criterion(pred_y, data_y)
        tot_loss.append(loss.item())
    model.to('cpu')
    return sum(tot_loss) / len(tot_loss)


def plot_2d_loss_landscape(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        caption: str,
        save_path: str,
        parameter_args: Dict = {"name": "all"},
        noise_range: Tuple[float, float] = (-1, 1),
        resolution: int = 20,
        visualize: bool = False,
        log_scale: bool = False
) -> None:
    """
    Overview:
        This is a function that use visualization techniques proposed in: Visualizing the Loss Landscape of Neural Nets.
        Currently, only linear layers and convolution layers will be considered.
    Arguments:
        model: The model that is needed to be checked for loss landscape.
        dataloader: The dataloader used to check the landscape.
        caption: The caption of generated graph.
        save_path: The save path of the generated loss landscape picture.
        parameter_args: Specify what parameters should add noises. Default to be ``{"name": "all"}``. For other \
            usages, please refer to the usage of ``aggregation_parameters`` in our configuration. A tutorial can \
            be found in: https://github.com/kxzxvbk/Fling/docs/meaning_for_configurations_en.md.
        device: The device to run on, such as ``"cuda"`` or ``"cpu"``.
        noise_range: The coordinate range of the loss-landscape, default to be ``(-1, 1)``.
        resolution: The resolution of generated landscape. A larger resolution will cost longer time for computation, \
            but a lower resolution may result in unclear contours. Default to be ``20``.
        visualize: Whether to directly show the picture in GUI. Default to be ``False``.
        log_scale: Whether to use a log function to normalize the loss. Default to be ``False``.
    """
    # Copy the original model.
    orig_model = model
    model = copy.deepcopy(model)

    # Generate two random directions.
    rand_x, rand_y = {}, {}
    for k, layer in model.named_modules():
        if parameter_args['name'] == 'all':
            incl = True
        elif parameter_args['name'] == 'contain':
            kw = parameter_args['keywords']
            incl = any([kk in k for kk in kw])
        elif parameter_args['name'] == 'except':
            kw = parameter_args['keywords']
            incl = all([kk not in k for kk in kw])
        else:
            raise ValueError(f"Illegal parameter_args: {parameter_args}")
        if not incl:
            continue

        if isinstance(layer, nn.Linear):
            orig_weight = copy.deepcopy(layer.weight)
            rand_x0 = _gen_rand_like(orig_weight)
            rand_y0 = _gen_rand_like(orig_weight)
        elif isinstance(layer, nn.Conv2d):
            orig_weight = copy.deepcopy(layer.weight)
            orig_weight = orig_weight.reshape(orig_weight.shape[0], -1)
            rand_x0 = _gen_rand_like(orig_weight)
            rand_y0 = _gen_rand_like(orig_weight)
        else:
            continue
        rand_x[k], rand_y[k] = rand_x0, rand_y0

    # Generate the meshgrid for loss landscape.
    x_coords = torch.linspace(noise_range[0], noise_range[1], resolution)
    y_coords = torch.linspace(noise_range[0], noise_range[1], resolution)
    loss_values = torch.zeros((resolution, resolution)).float()

    orig_layers = dict(orig_model.named_modules())
    with torch.no_grad():
        for i in tqdm(range(resolution)):
            for j in range(resolution):
                x_coord, y_coord = x_coords[i], y_coords[j]
                for k, layer in model.named_modules():
                    if k not in rand_x.keys():
                        continue
                    elif isinstance(layer, nn.Linear):
                        orig_weight = copy.deepcopy(orig_layers[k].weight)
                        orig_weight += rand_x[k] * x_coord + rand_y[k] * y_coord
                        layer.weight = orig_weight
                    elif isinstance(layer, nn.Conv2d):
                        orig_weight = copy.deepcopy(orig_layers[k].weight)
                        orig_shape = orig_weight.shape
                        orig_weight = orig_weight.reshape(orig_weight.shape[0], -1)
                        orig_weight += rand_x[k] * x_coord + rand_y[k] * y_coord
                        layer.weight.data = orig_weight.reshape(orig_shape)
                loss_values[i][j] = _calc_loss_value(model=model, data_loader=dataloader, device=device)
    if log_scale:
        loss_values = torch.log(loss_values)

    # Plot the result.
    x_mesh, y_mesh = torch.meshgrid(x_coords, y_coords)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_mesh, y_mesh, loss_values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title(caption)
    plt.savefig(save_path)
    if visualize:
        plt.show()
    plt.cla()
