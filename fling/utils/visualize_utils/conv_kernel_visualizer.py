import torchvision
from torch import nn

from fling.utils import Logger


def plot_conv_kernels(logger: Logger, layer: nn.Conv2d, name: str) -> None:
    """
    Overview:
        Plot the kernels in a certain convolution layer for better visualization.
    Arguments:
        logger: The logger to write result image.
        layer: The convolution layer to visualize.
        name: The name of the plotted figure.
    """
    param = layer.weight
    in_channels = param.shape[1]
    k_w, k_h = param.size()[3], param.size()[2]
    kernel_all = param.view(-1, 1, k_w, k_h)
    kernel_grid = torchvision.utils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
    logger.add_image(f'{name}', kernel_grid, global_step=0)
