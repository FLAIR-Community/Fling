import os
import torch
import random
from typing import Dict
from easydict import EasyDict

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler, get_launcher


def get_kwds(rnd: int, rnd2str: Dict) -> list:
    """
    Overview:
        Given the current index ``idx``, choose the function to decide the layers to train
    """
    rounds_per_key = 2
    total_len = len(list(rnd2str.keys()))

    idx = (rnd // rounds_per_key) % total_len
    kwds = rnd2str[idx]

    return kwds, idx


def get_train_args_transformer(rnd: int) -> Dict:
    """
    Overview:
        Given the current training round ``rnd``, decide the trainable parameters of a transformer model (3 encoder layers).
    """
    rnd2str = {
        0: ['_orig_mod.embedding.weight'], # 6066600
        1: ['_orig_mod.transformer_encoder.layers.0.self_attn'], # 160800
        2: ['_orig_mod.transformer_encoder.layers.0.linear', '_orig_mod.transformer_encoder.layers.0.norm'], # 1802
        3: ['_orig_mod.transformer_encoder.layers.1.self_attn'], # 160800
        4: ['_orig_mod.transformer_encoder.layers.1.linear', '_orig_mod.transformer_encoder.layers.1.norm'], # 1802
        5: ['_orig_mod.transformer_encoder.layers.2.self_attn'], # 160800
        6: ['_orig_mod.transformer_encoder.layers.2.linear', '_orig_mod.transformer_encoder.layers.2.norm'], # 1802
        7: ['_orig_mod.classifier.weight', '_orig_mod.classifier.bias'] # 1005
    }
    # FedPart(Repeat)
    if rnd < 5:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 5 and rnd < 21:
        kwds, idx = get_kwds(rnd - 5, rnd2str)
    elif rnd >= 21 and rnd < 26:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 26 and rnd < 42:
        kwds, idx = get_kwds(rnd - 26, rnd2str)
    elif rnd >= 42 and rnd < 47:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 47 and rnd < 63:
        kwds, idx = get_kwds(rnd - 47, rnd2str)
    elif rnd >= 63 and rnd < 68:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 68 and rnd < 84:
        kwds, idx = get_kwds(rnd - 68, rnd2str)
    elif rnd >= 84 and rnd < 89:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 89 and rnd < 105:
        kwds, idx = get_kwds(rnd - 89, rnd2str)
    return EasyDict({"name": "contain", "keywords": kwds}), idx


def get_train_args_resnet18(rnd: int) -> Dict:
    """
    Overview:
        Given the current training round ``rnd``, decide the trainable parameters of a ResNet-18 model.
    """
    # Total 197M (11227812)
    rnd2str = {
        0: ['pre_conv', 'pre_bn'],  # 9536
        1: ['layers.0.0.conv1', 'layers.0.0.bn1'],  # 36992
        2: ['layers.0.0.conv2', 'layers.0.0.bn2'],  # 36992
        3: ['layers.0.1.conv1', 'layers.0.1.bn1'],  # 36992
        4: ['layers.0.1.conv2', 'layers.0.1.bn2'],  # 36992

        5: ['layers.1.0.conv1', 'layers.1.0.bn1'],  # 73984
        6: ['layers.1.0.conv2', 'layers.1.0.bn2'],  # 147712
        7: ['layers.1.0.downsample.0', 'layers.1.0.downsample.1'],  # 8448
        8: ['layers.1.1.conv1', 'layers.1.1.bn1'],  # 147712
        9: ['layers.1.1.conv2', 'layers.1.1.bn2'],  # 147712

        10: ['layers.2.0.conv1', 'layers.2.0.bn1'],  # 295424
        11: ['layers.2.0.conv2', 'layers.2.0.bn2'],  # 590336
        12: ['layers.2.0.downsample.0', 'layers.2.0.downsample.1'],  # 33280
        13: ['layers.2.1.conv1', 'layers.2.1.bn1'],  # 590336
        14: ['layers.2.1.conv2', 'layers.2.1.bn2'],  # 590336

        15: ['layers.3.0.conv1', 'layers.3.0.bn1'],  # 1180672
        16: ['layers.3.0.conv2', 'layers.3.0.bn2'],  # 2360320
        17: ['layers.3.0.downsample.0', 'layers.3.0.downsample.1'],  # 132096
        18: ['layers.3.1.conv1', 'layers.3.1.bn1'],  # 2360320
        19: ['layers.3.1.conv2', 'layers.3.1.bn2'],  # 2360320

        20: ['fc']  # 51300
    }
    # FedPart(Repeat)
    if rnd < 5:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 5 and rnd < 47:
        kwds, idx = get_kwds(rnd - 5, rnd2str)
    elif rnd >= 47 and rnd < 52:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 52 and rnd < 94:
        kwds, idx = get_kwds(rnd - 52, rnd2str)
    elif rnd >= 94 and rnd < 99:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 99 and rnd < 141:
        kwds, idx = get_kwds(rnd - 99, rnd2str)
    return EasyDict({"name": "contain", "keywords": kwds}), idx


def get_train_args_resnet8(rnd: int) -> Dict:
    """
    Overview:
        Given the current training round ``rnd``, decide the trainable parameters of a ResNet-8 model.
    """
    # Total 197M
    rnd2str = {
        0: ['pre_conv', 'pre_bn'],  # 1.5
        1: ['layers.0.0.conv1', 'layers.0.0.bn1'],  # 5.898
        2: ['layers.0.0.conv2', 'layers.0.0.bn2'],  # 5.898
        3: ['layers.1.0.conv1', 'layers.1.0.bn1'],  # 11.8
        4: ['layers.1.0.conv2', 'layers.1.0.bn2'],  # 23.59
        5: ['layers.1.0.downsample.0', 'layers.1.0.downsample.1'],  # 1.311
        6: ['layers.2.0.conv1', 'layers.2.0.bn1'],  # 47.19
        7: ['layers.2.0.conv2', 'layers.2.0.bn2'],  # 94.37
        8: ['layers.2.0.downsample.0', 'layers.2.0.downsample.1'],  # 5.243
        9: ['fc']  # 0.411
    }
    # FedPart(Repeat)
    if rnd < 5:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 5 and rnd < 25:
        kwds, idx = get_kwds(rnd - 5, rnd2str)
    elif rnd >= 25 and rnd < 30:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 30 and rnd < 50:
        kwds, idx = get_kwds(rnd - 30, rnd2str)
    elif rnd >= 50 and rnd < 55:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 55 and rnd < 75:
        kwds, idx = get_kwds(rnd - 55, rnd2str)
    elif rnd >= 75 and rnd < 80:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 80 and rnd < 100:
        kwds, idx = get_kwds(rnd - 80, rnd2str)
    elif rnd >= 100 and rnd < 105:
        return EasyDict({"name": "all"}), -1
    elif rnd >= 105 and rnd < 125:
        kwds, idx = get_kwds(rnd - 105, rnd2str)
    return EasyDict({"name": "contain", "keywords": kwds}), idx


def get_train_args(rnd: int, model_name: str) -> Dict:
    if model_name == 'resnet8':
        return get_train_args_resnet8(rnd)
    elif model_name == 'resnet18':
        return get_train_args_resnet18(rnd)
    elif model_name == 'transformer_classifier':
        return get_train_args_transformer(rnd)


def get_aggr_args(rnd: int, model_name: str) -> Dict:
    if model_name == 'resnet8':
        tmp = get_train_args_resnet8(rnd)
    elif model_name == 'resnet18':
        tmp = get_train_args_resnet18(rnd)
    elif model_name == 'transformer_classifier':
        tmp = get_train_args_transformer(rnd)
    return tmp


def partial_model_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for partial federated learning. In each training round, a part of the model can be selected in order \
       from shallow to deep to be trained and aggregated. The final performance of is calculated by averaging the \
       local model in each client. Typically, each local model is tested using local test dataset.
       This pipeline is the base implementation of FedPart introduced in: 
       Why Go Full? Elevating Federated Learning Through Partial Network Updates
       <link https://arxiv.org/abs/2410.11559 link>.
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed=seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset.
    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)
    # Split dataset into clients.
    train_sets = data_sampling(train_set, args, seed, train=True)
    test_sets = data_sampling(test_set, args, seed, train=False)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)
    for i in range(args.client.client_num):
        group.append(get_client(args=args, client_id=i, train_dataset=train_sets[i], test_dataset=test_sets[i]))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.optimizer.lr, args=args.learn.scheduler)

    # Setup launcher.
    launcher = get_launcher(args)

    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        train_args, index = get_train_args(rnd=i, model_name=args.model.name)
        aggr_args, _ = get_aggr_args(rnd=i, model_name=args.model.name)
        
        # Initialize variable monitor.
        train_monitor = VariableMonitor()

        # Random sample participated clients in each communication round.
        participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)
    
        # Local training for each participated client and add results to the monitor.
        # Use multiprocessing for acceleration.
        train_results = launcher.launch(
            clients=[group.clients[j] for j in participated_clients],
            lr=cur_lr, task_name='train', train_args=train_args
        )

        for item in train_results:
            train_monitor.append(item)

        # Testing
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)

            # Logging test variables.
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=test_result, rnd=i)

            # Saving model checkpoints.
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))

        # Aggregate parameters in each client.
        trans_cost = group.aggregate(i, aggr_parameter_args=aggr_args)

        # Logging for train variables.
        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update({'trans_cost(MB)': trans_cost / 1e6, 'lr': cur_lr, 'index': int(index)})
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)

        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=test_result, rnd=i)

            # Saving model checkpoints.
            torch.save(group.clients[0].model.state_dict(), os.path.join(args.other.logging_path, 'model.ckpt'))