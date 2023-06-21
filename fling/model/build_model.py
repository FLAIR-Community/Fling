import copy
import platform
import warnings

import torch

from fling.utils.registry_utils import MODEL_REGISTRY


def get_model(args):
    args = copy.deepcopy(args)
    model_name = args.model.pop('name')
    model = MODEL_REGISTRY.build(model_name, **args.model)
    if torch.__version__[0] == '2':
        if platform.system().lower() == 'linux':
            warnings.warn('Using PyTorch >= 2.0, compiling the model ...')
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model)
        else:
            warnings.warn('Using PyTorch >= 2.0, but current platform is: '
                          + platform.system() + '  Give up compiling...')
        return model
