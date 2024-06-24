import copy
import math
from typing import Tuple, Dict, Optional
from matplotlib import pyplot as plt
from tqdm import tqdm
from fling.utils.torch_utils import model_add, model_sub, model_mul

import torch
from torch import nn
from torch.utils.data import DataLoader


def _gen_rand_like(tensor: torch.Tensor) -> torch.Tensor:
    # Return a tensor whose shape is identical to the input tensor.
    # The returned tensor is a filled with Gaussian noise and the norm in each line is the same as the input.
    tmp = torch.randn_like(tensor)
    return tmp * torch.norm(tensor, dim=1, keepdim=True) / torch.norm(tmp, dim=1, keepdim=True)


def _calc_loss_value(
    model: nn.Module, data_loader: DataLoader, device: str, criterion: nn.Module = nn.CrossEntropyLoss()
):
    # Given a model and corresponding dataset, calculate the mean loss value.
    model = model.to(device)
    model.eval()
    tot_loss = []
    for _, (data) in enumerate(data_loader):
        data_x, data_y = data['input'].to(device), data['class_id'].to(device)
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
    target_model1: Optional[nn.Module] = None,
    target_model2: Optional[nn.Module] = None,
    parameter_args: Dict = {"name": "all"},
    noise_range: Tuple[float, float] = (-1, 1),
    resolution: int = 20,
    visualize: bool = False,
    log_scale: bool = False,
    max_val: float = 5,
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
        target_model1: If specified, the first direction of visualization will be not randomly chosen, but will be set \
            as ``target_model1 - model``.
        target_model2: Similar to ``target_model1``, determine the second direction of visualization.
        parameter_args: Specify what parameters should add noises. Default to be ``{"name": "all"}``. For other \
            usages, please refer to the usage of ``aggregation_parameters`` in our configuration. A tutorial can \
            be found in: https://github.com/FLAIR-Community/Fling/docs/meaning_for_configurations_en.md.
        device: The device to run on, such as ``"cuda"`` or ``"cpu"``.
        noise_range: The coordinate range of the loss-landscape, default to be ``(-1, 1)``.
        resolution: The resolution of generated landscape. A larger resolution will cost longer time for computation, \
            but a lower resolution may result in unclear contours. Default to be ``20``.
        visualize: Whether to directly show the picture in GUI. Default to be ``False``.
        log_scale: Whether to use a log function to normalize the loss. Default to be ``False``.
        max_val: The max value of permitted loss. This is for better visualization.
    """
    # Copy the original model.
    orig_model = model
    model = copy.deepcopy(model)
    model.eval()

    # Generate two random directions.
    rand_x, rand_y = {}, {}
    for k, layer in model.named_modules():
        # Decide which parameters should be included.
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

        # Generate random noises.
        if (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d)) \
                and target_model1 is not None and target_model2 is not None:
            # Detail: If the target models are specified, all parameters together with statistics (e.g. BN statistics)
            # will be permuted. If the target models are not specified, only weight tensors in linear layers and
            # convolution layers will be permuted.
            rand_x0 = model_sub(dict(target_model1.named_modules())[k], layer)
            rand_y0 = model_sub(dict(target_model2.named_modules())[k], layer)
        elif isinstance(layer, nn.Linear):
            # Generate random linear weight tensors.
            rand_x0 = _gen_rand_like(layer.weight)
            rand_y0 = _gen_rand_like(layer.weight)
        elif isinstance(layer, nn.Conv2d):
            # Generate random convolution weight tensors.
            orig_weight = layer.weight.reshape(layer.weight.shape[0], -1)
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
                    elif target_model1 is not None and target_model2 is not None:
                        # If the target models are specified, manipulate the total model.
                        new_layer = model_add(model_mul(x_coord, rand_x[k]), model_mul(y_coord, rand_y[k]))
                        new_layer = model_add(new_layer, orig_layers[k])
                        # Copy the new generated layers to the original object.
                        layer.load_state_dict(new_layer.state_dict())
                    elif isinstance(layer, nn.Linear):
                        # Target models are not specified, only operate the weight tensors.
                        orig_weight = orig_layers[k].weight.clone()
                        delta_w = rand_x[k] * x_coord + rand_y[k] * y_coord
                        orig_weight += delta_w
                        layer.weight.data = orig_weight
                    elif isinstance(layer, nn.Conv2d):
                        # Operate on the convolution weight tensors.
                        orig_weight = orig_layers[k].weight.clone()
                        orig_shape = orig_weight.shape
                        orig_weight = orig_weight.reshape(orig_weight.shape[0], -1)
                        delta_w = rand_x[k] * x_coord + rand_y[k] * y_coord
                        orig_weight += delta_w
                        layer.weight.data = orig_weight.reshape(orig_shape)
                loss_values[i][j] = min(_calc_loss_value(model=model, data_loader=dataloader, device=device), max_val)
    if log_scale:
        loss_values = torch.log(loss_values)

    # Plot the result.
    x_mesh, y_mesh = torch.meshgrid(x_coords, y_coords)
    ax = plt.axes(projection='3d')

    # Add special dots.
    non_nan_tensor = loss_values[~torch.isnan(loss_values)]
    max_loss = torch.max(non_nan_tensor)

    loss1 = _calc_loss_value(model=orig_model, data_loader=dataloader, device=device)
    if log_scale:
        loss1 = math.log(loss1)
    ax.text(0, 0, max_loss, "GM ({:.2f})".format(loss1), color='black', zorder=2)

    # Add client model dots.
    if target_model1 is not None and target_model2 is not None:
        loss1 = _calc_loss_value(model=target_model1, data_loader=dataloader, device=device)
        if log_scale:
            loss1 = math.log(loss1)
        ax.text(1, 1, max_loss, "LM1 ({:.2f})".format(loss1), color='black', zorder=2)

        loss2 = _calc_loss_value(model=target_model2, data_loader=dataloader, device=device)
        if log_scale:
            loss2 = math.log(loss2)
        ax.text(0, 1, max_loss, "LM2 ({:.2f})".format(loss2), color='black', zorder=2)

    ax.plot_surface(x_mesh, y_mesh, loss_values, rstride=1, cstride=1, cmap='viridis', edgecolor='none', zorder=1)
    ax.set_title(caption)
    plt.savefig(save_path)
    if visualize:
        plt.show()
    plt.close()
