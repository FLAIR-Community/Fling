from torchvision.models import resnet18

from fling.utils import Logger
from fling.utils.visualize_utils import plot_conv_kernels

if __name__ == '__main__':
    # Step 1: prepare the model.
    model = resnet18(pretrained=True)

    # Step 2: prepare the logger.
    logger = Logger('resnet18_conv_kernels')

    # Step 3: save the kernels.
    plot_conv_kernels(logger, model.conv1, name='pre-conv')
