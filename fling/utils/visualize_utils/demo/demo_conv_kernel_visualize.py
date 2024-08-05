from torchvision.models import resnet18

from fling.utils import Logger
from fling.utils.visualize_utils import plot_conv_kernels
from easydict import EasyDict
from fling.utils.registry_utils import MODEL_REGISTRY

if __name__ == '__main__':
    # Step 1: prepare the model.
    model_arg = EasyDict(dict(
        name='resnet8',
        input_channel=3,
        class_number=100,
    ))
    model_name = model_arg.pop('name')
    model = MODEL_REGISTRY.build(model_name, **model_arg)

    # You can also initialize the model without using configurations.
    # e.g. model = resnet18(pretrained=True)

    # Step 2: prepare the logger.
    logger = Logger('resnet18_conv_kernels')

    # Step 3: save the kernels.
    plot_conv_kernels(logger, layer=model.conv1, name='pre-conv')
