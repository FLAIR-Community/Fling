import os
import torch
from typing import Dict
from easydict import EasyDict

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler, get_launcher


def get_train_args_resnet_v1(rnd: int) -> Dict:
    """
    Overview:
        Given the current training round ``rnd``, decide the trainable parameters.
    """
    # Total 197M
    rnd2str = {
        0: ['pre_conv'],  # 1.5
        1: ['layers.0.0.conv1', 'layers.0.0.bn1'],  # 5.898
        2: ['layers.0.0.conv2', 'layers.0.0.bn2'],  # 5.898
        3: ['layers.1.0.conv1', 'layers.1.0.bn1'],  # 11.8
        4: ['layers.1.0.conv2', 'layers.1.0.bn2'],  # 23.59
        5: ['layers.1.0.downsample'],  # 1.311
        6: ['layers.2.0.conv1', 'layers.2.0.bn1'],  # 47.19
        7: ['layers.2.0.conv2', 'layers.2.0.bn2'],  # 94.37
        8: ['layers.2.0.downsample'],  # 5.243
        9: ['fc']  # 0.411
    }
    rounds_per_key = 2
    total_len = len(list(rnd2str.keys()))
    if rnd < 5:
        return EasyDict({"name": "all"})
    rnd -= 5
    print('index: ' + str((rnd // rounds_per_key) % total_len))
    return EasyDict({"name": "contain", "keywords": rnd2str[(rnd // rounds_per_key) % total_len]})


def get_train_args_resnet(rnd: int) -> Dict:
    return get_train_args_resnet_v1(rnd)


def get_aggr_args_resnet(rnd: int) -> Dict:
    tmp = get_train_args_resnet_v1(rnd)
    return tmp


def get_train_args(rnd: int) -> Dict:
    """
    Overview:
        Given the current training round ``rnd``, decide the trainable parameters.
    """
    rnd2str = {
        0: ['layers.0'],
        1: ['layers.2'],
        2: ['layers.4'],
        3: ['fc.1']
    }
    rounds_per_key = 2
    total_len = len(list(rnd2str.keys()))
    if rnd < 5:
        return EasyDict({"name": "all"})
    rnd -= 5
    return EasyDict({"name": "contain", "keywords": rnd2str[(rnd // rounds_per_key) % total_len]})


def get_aggr_args(rnd: int) -> Dict:
    """
    Overview:
        Given the current training round ``rnd``, decide the aggregation parameters.
    """
    return get_train_args(rnd)


def partial_model_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for partial federated learning. In each training round, a part of the model will be trained and \
       aggregated. The final performance of is calculated by averaging the local model in each client. Typically, \
       each local model is tested using local test dataset.
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
    lr_scheduler = LRScheduler(args)

    # Setup launcher.
    launcher = get_launcher(args)

    # Training loop
    n_iteration = 0
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        train_args = get_train_args_resnet(rnd=i)
        aggr_args = get_aggr_args_resnet(rnd=i)

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
            lr=cur_lr, task_name='train', train_args=train_args, ret_full_log=True
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
        mean_train_variables.update({'trans_cost(MB)': trans_cost / 1e6, 'lr': cur_lr})
        for k in mean_train_variables.keys():
            item = mean_train_variables[k]
            if isinstance(item, float):
                logger.add_scalar(f'train/{k}', item, i)
            else:
                length = len(item)
                for j in range(length):
                    logger.add_scalar(f'train/{k}', item[j], n_iteration + j)
        n_iteration += length

        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=test_result, rnd=i)

            # Saving model checkpoints.
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))

    # Fine-tuning
    # Fine-tune model on each client and collect all the results.
    finetune_results = launcher.launch(
        clients=[group.clients[j] for j in range(args.client.client_num)],
        lr=cur_lr,
        finetune_args=args.learn.finetune_parameters,
        task_name='finetune'
    )

    # Logging fine-tune results
    for key in finetune_results[0][0].keys():
        for eid in range(len(finetune_results[0])):
            tmp_mean = sum([finetune_results[cid][eid][key]
                            for cid in range(len(finetune_results))]) / len(finetune_results)
            logger.add_scalar(f'finetune/{key}', tmp_mean, eid)