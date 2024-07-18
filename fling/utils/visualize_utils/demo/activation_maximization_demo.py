import os
from easydict import EasyDict

from fling.utils.registry_utils import MODEL_REGISTRY
from fling.utils.visualize_utils import ActivationMaximizer


def am_demo(path_head, model, layer, channel_id):
    working_dir = os.path.join(path_head, layer)
    activation_maximizer = ActivationMaximizer(iteration=1000, working_dir=working_dir, tv_weight=1)
    activation_maximizer.activation_maximization(
        model, layer, channel_id=channel_id, image_shape=[3, 32, 32], device='cuda', learning_rate=1e-1
    )


if __name__ == '__main__':
    # path to store the result
    name = 'demo'
    out_dir = os.path.join('./visualize/', name)

    # Step 1: prepare the model.
    model_arg = EasyDict(dict(
        name='resnet8',
        input_channel=3,
        class_number=100,
    ))
    model_name = model_arg.pop('name')
    model = MODEL_REGISTRY.build(model_name, **model_arg)
    print(model)

    # Step 2: Loop to perform am on each layer of the model.
    layers = [
        'conv1', 'layers.0.0.conv1', 'layers.0.0.conv2', 'layers.1.0.conv1', 'layers.1.0.conv2',
        'layers.1.0.downsample.0', 'layers.2.0.conv1', 'layers.2.0.conv2', 'layers.2.0.downsample.0', 'fc'
    ]
    for layer in layers:
        for channel_id in range(64):
            am_demo(out_dir, model, layer, channel_id)
