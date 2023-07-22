import copy
import platform
import warnings

import torch

from fling.utils.registry_utils import MODEL_REGISTRY


def get_model(args: dict) -> torch.nn.Module:
    # Copy the args to prevent it from modified by ``args.pop('xxx')``
    args = copy.deepcopy(args)
    # Get the model constructed by args.
    model_name = args.model.pop('name')
    model = MODEL_REGISTRY.build(model_name, **args.model)
    # Check the PyTorch version and current platform.
    # If the version is greater than 2.0.0 and the current platform is Linux, using the compiling mode for efficiency.
    if torch.__version__[0] == '2' and False:
        # Compile
        if platform.system().lower() == 'linux':
            warnings.warn('Using PyTorch >= 2.0, compiling the model ...')
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model)
        # Non-compile
        else:
            warnings.warn(
                'Using PyTorch >= 2.0, but current platform is: ' + platform.system() + '  Give up compiling...'
            )
    # Non-compile
    else:
        warnings.warn(f'Using PyTorch version: {torch.__version__}, skip compiling...')
    return model
