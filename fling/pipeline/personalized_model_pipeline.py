import os
import tqdm
import torch

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler


def personalized_model_serial_pipeline(args, seed=0):
    args = compile_config(args, seed=seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset.
    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)
    # Split dataset into clients.
    train_sets = data_sampling(train_set, args)

    # Initialize clients, assemble datasets.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)
    for i in range(args.client.client_num):
        group.append(get_client(train_sets[i], args=args, client_id=i))
    group.initialize()

    # training loop
    lr_scheduler = LRScheduler(args)
    for i in range(args.learn.global_eps):
        train_monitor = VariableMonitor(['train_acc', 'train_loss'])
        logger.logging('Starting round: ' + str(i))

        # Random sample participated clients in each communication round.
        participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)
        for j in tqdm.tqdm(participated_clients):
            train_monitor.append(group.clients[j].train(lr=cur_lr))

        # Aggregation and sync.
        trans_cost = group.aggregate(i, tb_logger=logger)

        # Logging
        mean_train_variables = train_monitor.variable_mean()
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)
        extra_info = {'trans_cost': trans_cost / 1e6, 'lr': cur_lr}
        logger.add_scalars_dict(prefix='train', dic=extra_info, rnd=i)

        if i % args.other.test_freq == 0:
            test_monitor = VariableMonitor(['test_acc', 'test_loss'])
            for j in range(args.client.client_num):
                test_monitor.append(group.clients[j].test())
            mean_test_variables = test_monitor.variable_mean()
            logger.add_scalars_dict(prefix='test', dic=mean_test_variables, rnd=i)
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))

    # Finetune.
    finetune_results = [
        group.clients[i].finetune(lr=cur_lr, finetune_args=args.learn.finetune_parameters)
        for i in range(len(group.clients))
    ]

    for key in finetune_results[0][0].keys():
        for eid in range(len(finetune_results[0])):
            tmp_mean = sum([finetune_results[cid][eid][key]
                            for cid in range(len(finetune_results))]) / len(finetune_results)
            logger.add_scalar(f'finetune/{key}', tmp_mean, eid)
