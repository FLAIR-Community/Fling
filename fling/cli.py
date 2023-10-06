import os
import importlib
import pickle
from typing import Iterable, Dict, List
import warnings
import click
from click import Context, Option
from copy import deepcopy
from prettytable import PrettyTable

from fling import __TITLE__, __VERSION__

COMMAND_FILE = './cli.tmp'  # File to save defined commands.
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def has_nested_attr(obj: object, attr_str: str) -> bool:
    # This is for recursively attribute-seeking.
    attrs = attr_str.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    return hasattr(obj, attrs[-1])


def set_nested_attr(obj: object, attr_str: str, value: object) -> None:
    # This is for recursively attribute-setting.
    attrs = attr_str.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


def print_version_callback(ctx: Context, param: Option, value: bool) -> None:
    # Callback function for -v option.
    if not value or ctx.resilient_parsing:
        return
    click.echo('{title}, version {version}.'.format(title=__TITLE__, version=__VERSION__))
    ctx.exit()


def add_arguments_callback(ctx: Context, param: Option, values: Iterable) -> Dict:
    # Callback function for --argument_map option.
    result = {}
    for value in values:
        key, val = value.split(":", 1)
        result[key] = val
    return result


def extra_arguments_callback(ctx: Context, param: Option, values: Iterable) -> Dict:
    # Callback function for --extra_argument option.
    result = {}
    for value in values:
        key, val = value.split(":", 1)
        result[key] = val
    return result


def seed_callback(ctx: Context, param: Option, values: str) -> List:
    # Callback function for --seed option.
    seeds = values.strip().split(',')
    return [int(s.strip()) for s in seeds]


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    '-v',
    '--version',
    is_flag=True,
    callback=print_version_callback,
    expose_value=False,
    is_eager=True,
    help="Show package's version information."
)
@click.option('-pc', '--print_config', is_flag=True, help="Whether to print config in the command line.")
@click.argument('mode', type=str)
# arguments for: fling run
@click.option('-s', '--seed', type=str, help='Seeds number. Usage: --seed 0,1,2', default='0', callback=seed_callback)
@click.option('-c', '--config', type=str, help='Path for config file')
@click.option(
    '-e',
    '--extra_argument',
    multiple=True,
    callback=extra_arguments_callback,
    help='Usage: --extra_argument key1:value1 --extra_argument key2:value2'
)
@click.option('-p', '--pipeline', type=str, help='The pipeline function of this command.')
# arguments for: fling create / remove
@click.option('-n', '--name', type=str, help='Command name to be created')
@click.option(
    '-a',
    '--argument_map',
    multiple=True,
    callback=add_arguments_callback,
    help='Usage: --argument_map key1:value1 --argument_map key2:value2'
)
def cli(
    mode: str,
    seed: List,
    print_config: bool,
    config: str,
    extra_argument: Dict,
    name: str,
    pipeline: str,
    argument_map: Dict,
):
    # fling create xxx
    if mode == 'create':
        return create_command(name, argument_map)

    # fling remove xxx
    if mode == 'remove':
        return remove_command(name)

    # fling list
    if mode == 'list':
        return list_command()

    # fling info
    if mode == 'info':
        return command_info(name)

    # fling run xxx
    if mode == 'run':
        if config.endswith('.py'):
            config = config[:-3]
        base_args = getattr(importlib.import_module(config.replace('/', '.')), 'exp_args')
        if print_config:
            base_args.other.print_config = True

        for k, v in extra_argument.items():
            v = auto_convert(v)
            if not has_nested_attr(base_args, k):
                warnings.warn(f"Can not find key: {k} in the config file.")
            set_nested_attr(base_args, k, v)

        pipeline_func = getattr(importlib.import_module('fling.pipeline'), pipeline)
        for s in seed:
            pipeline_func(args=deepcopy(base_args), seed=s)
        return

    # Default process.
    # To deal with self-defined commands.
    with open(COMMAND_FILE, 'rb') as f:
        commands = pickle.load(f)
    if mode not in commands.keys():
        raise ValueError(f'Unrecognized command mode: {mode}')

    # Import the base arg file.
    if config.endswith('.py'):
        config = config[:-3]
    base_args = getattr(importlib.import_module(config.replace('/', '.')), 'exp_args')
    if print_config:
        base_args.other.print_config = True

    # Update the base arg file using arguments passed in.
    argument_map = commands[mode]
    for k, v in extra_argument.items():
        if k not in argument_map.keys() and not has_nested_attr(base_args, k):
            warnings.warn(f'The argument {k} is not defined in command {mode}, and neither in the config file.')
        v = auto_convert(v)
        dst_key = k if k not in argument_map.keys() else argument_map[k]
        set_nested_attr(base_args, dst_key, v)

    # Get the pipeline function.
    pipeline_func = getattr(importlib.import_module('fling.pipeline'), pipeline)
    for s in seed:
        pipeline_func(args=deepcopy(base_args), seed=s)


def create_command(name: str, add_arguments: Iterable):
    # Create a new command and add it into the command file.
    # Check whether the name is the same as built-in names.
    if name in ['run', 'remove', 'list', 'create', 'info']:
        raise ValueError(f'You are not supposed to define a command named {name}. Try another one.')

    # If the command file database already exists, load the original file and modify it.
    # Otherwise, create a new command file database.
    if os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, 'rb') as f:
            orig_command_dict = pickle.load(f)
        if name in orig_command_dict.keys():
            raise KeyError(
                'Current command name is already defined. Please use `fling list` to show all defined '
                'commands.'
            )
        orig_command_dict[name] = add_arguments
    else:
        orig_command_dict = {name: add_arguments}

    with open(COMMAND_FILE, 'wb') as f:
        pickle.dump(orig_command_dict, f)


def remove_command(name: str):
    # Read the command database.
    if not os.path.exists(COMMAND_FILE):
        commands = {}
    else:
        with open(COMMAND_FILE, 'rb') as f:
            commands = pickle.load(f)

    # Remove the command.
    if name not in commands:
        raise KeyError(f'The command {name} is never defined. Please use `fling list` to show all defined commands.')
    commands.pop(name)

    # Write back.
    with open(COMMAND_FILE, 'wb') as f:
        pickle.dump(commands, f)


def list_command():
    # Read the command database.
    if not os.path.exists(COMMAND_FILE):
        commands = {}
    else:
        with open(COMMAND_FILE, 'rb') as f:
            commands = pickle.load(f)

    click.echo("Defined commands: \n" + ', '.join(list(commands.keys())))


def command_info(name: str):
    # Read the command database.
    if not os.path.exists(COMMAND_FILE):
        commands = {}
    else:
        with open(COMMAND_FILE, 'rb') as f:
            commands = pickle.load(f)

    # Remove the command.
    if name not in commands:
        raise KeyError(f'The command {name} is never defined. Please use `fling list` to show all defined commands.')
    arg_map = commands[name]

    tb = PrettyTable(["Mapping key", "Mapping destination"])
    for k, v in arg_map.items():
        tb.add_row([k, v])

    click.echo(tb)


def auto_convert(var: str):
    # Auto conversion.
    try:
        return eval(var)
    except Exception:
        return var
