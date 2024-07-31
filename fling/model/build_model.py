import copy
import platform
import warnings

import torch

from fling.utils.registry_utils import MODEL_REGISTRY


def get_model(args: dict) -> torch.nn.Module:
    # Copy the args to prevent it from modified by ``args.pop('xxx')``
    args = copy.deepcopy(args)
    # Avoid bugs for PyTorch with lower versions.
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        warnings.warn('Fail to set: torch.set_float32_matmul_precision("high")')
    # Get the model constructed by args.
    model_name = args.model.pop('name')
    model = MODEL_REGISTRY.build(model_name, **args.model)

    # Choose whether to compile the model.
    # Check the launcher type.
    # Currently, compiled model does not support multiprocess training.
    # The model will be compiled only when the launcher type is serial.
    if args.launcher.name == 'serial':
        # Check the PyTorch version and current platform.
        # If the version is greater than 2.0.0 and the current platform is Linux,
        # using the compiling mode for better efficiency.
        if torch.__version__[0] == '2':
            # Compile
            if platform.system().lower() == 'linux':
                warnings.warn('Using PyTorch >= 2.0, compiling the model ...')
                try:
                    model = torch.compile(model)
                except Exception as e:
                    warnings.warn(f'Compile error, skip compiling...')
            # Non-compile
            else:
                warnings.warn(
                    'Using PyTorch >= 2.0, but current platform is: ' + platform.system() + ', give up compiling...'
                )
        # Non-compile
        else:
            warnings.warn(f'Using PyTorch version: {torch.__version__}, skip compiling...')
    # Non-compile
    else:
        warnings.warn(f'Trying to use launcher type: {args.launcher.name}, skip compiling...')

    return model
