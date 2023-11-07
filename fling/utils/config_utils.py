import os
import copy
import warnings
from easydict import EasyDict

import torch.multiprocessing as mp

from flzoo.default_config import default_exp_args
from fling.utils import seed_everything
from fling.utils.registry_utils import DATASET_REGISTRY


def save_config_file(config: dict, path: str) -> None:
    """
    Overview:
        Save configuration to python file
    Arguments:
        - config: Config dict
        - path: Path of saved file
    """
    config_string = str(config)
    from yapf.yapflib.yapf_api import FormatCode
    config_string, _ = FormatCode(config_string)
    config_string = config_string.replace('inf,', 'float("inf"),')
    with open(path, "w") as f:
        f.write('exp_config = ' + config_string)
    if config.other.print_config:
        print('exp_config = ' + config_string + '\n')


def compile_config(new_config: dict, seed: int) -> dict:
    r"""
    Overview:
        This function includes some important steps before the main process starts:
        1) Set the random seed for reproducibility.
        2) Determine the multiprocessing backend.
        3) Merge config (user config & default config).
        4) Compile data augmentation config.
        5) Create logging path and save the compiled config.
    Arguments:
        new_config: user-defined config.
        seed: random seed.
    Returns:
        result_config: the compiled config diction.
    """
    # Set random seed.
    seed_everything(seed)
    # Determine the multiprocessing backend.
    mp.set_start_method('spawn', force=True)

    merged_config = deep_merge_dicts(default_exp_args, new_config)
    compile_data_augmentation_config(merged_config)
    result_config = EasyDict(merged_config)

    # Create logging path and save the compiled config.
    exp_dir = result_config.other.logging_path
    if not os.path.exists(exp_dir):
        try:
            os.makedirs(exp_dir)
        except FileExistsError:
            warnings.warn("Logging directory already exists.")
    save_config_file(result_config, os.path.join(exp_dir, 'total_config.py'))

    return result_config


def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        Merge two dicts by calling ``deep_update``. The key of ``original`` dict will be updated by ``new_dict``.
        This is a recursive function.
    Arguments:
        - original: The dict to be updated.
        - new_dict: the dict to update ``original``,
    Returns:
        - merged_dict: A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])
    return merged


def deep_update(
        original: dict,
        new_dict: dict,
        new_keys_allowed: bool = False,
        whitelist=None,
        override_all_if_type_changes=None
) -> dict:
    r"""
    Overview:
        Update original dict with values from new_dict recursively.
    Arguments:
        - original: Dictionary with default values.
        - new_dict: Dictionary with values to be updated
        - new_keys_allowed: Whether new keys are allowed.
        - whitelist:
            List of keys that correspond to dict
            values where new sub-keys can be introduced. This is only at the top
            level.
        - override_all_if_type_changes:
            List of top level
            keys with value=dict, for which we always simply override the
            entire value (:obj:`dict`), if the "type" key in that value dict changes.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise RuntimeError("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new sub-keys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def compile_data_augmentation_config(cfg: dict) -> None:
    r"""
    Overview:
        This is an in-place operation that compile the data augmentation part of the configuration.
        If ``include_default=True``, the default data augmentations of each dataset will be applied.
    Arguments:
        cfg: The configuration file to be compiled.
    """
    if 'include_default' not in cfg['data']['transforms']:
        return
    include_default = cfg['data']['transforms'].pop('include_default')
    if include_default:
        dataset_module = DATASET_REGISTRY.get(cfg['data']['dataset'])
        if 'default_augmentation' in dataset_module.__dict__:
            default_cfg = dataset_module.default_augmentation
        else:
            default_cfg = dict()
        cfg['data']['transforms'] = deep_update(default_cfg, cfg['data']['transforms'])
