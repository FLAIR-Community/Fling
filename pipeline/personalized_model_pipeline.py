import os
import tqdm
import torch

from component.client import Client
from component.server import Server
from component.group import ParameterServerGroup
from utils.data_utils import DatasetConstructor, data_sampling
from utils import Logger, seed_everything, compile_config, client_sampling, VariableMonitor


def personalized_model_serial_pipeline(args, seed=0):
    seed_everything(seed)
    args = compile_config(args)

    # Construct logger.
    logger = Logger(args.logging_path)

    # Load dataset.
    train_set = DatasetConstructor(args).get_dataset()
    test_set = DatasetConstructor(args).get_dataset(train=False)
    # Split dataset into clients.
    train_sets = data_sampling(train_set, args)

    # Initialize clients, assemble datasets.
    group = ParameterServerGroup(args, logger)
    group.server = Server(args, args.device, test_dataset=test_set)
    for i in range(args.client_num):
        group.append(Client(train_sets[i], args=args, client_id=i))
    group.initialize()

    # training loop
    for i in range(args.glob_eps):
        train_monitor = VariableMonitor(['train_acc', 'train_loss'])
        logger.logging('Starting round: ' + str(i))

        # Random sample participated clients in each communication round.
        participated_clients = client_sampling(range(args.client_num), args.client_sample_rate)

        # Adjust learning rate.
        cur_lr = args.lr * (args.decay_factor ** i)
        for j in tqdm.tqdm(participated_clients):
            train_monitor.append(group.clients[j].train(lr=cur_lr))

        if i % args.test_freq == 0:
            test_monitor = VariableMonitor(['acc', 'loss'])
            for j in range(args.client.client_num):
                test_monitor.append(group.clients[j].test())
            mean_test_variables = test_monitor.variable_mean()
            for k in mean_test_variables:
                logger.add_scalar('test_before_aggregation/{}'.format(k), mean_test_variables[k], i)

        # Aggregation and sync.
        trans_cost = group.aggregate(i, tb_logger=logger)

        # Logging
        mean_train_variables = train_monitor.variable_mean()
        for k in mean_train_variables:
            logger.add_scalar('train/{}'.format(k), mean_train_variables[k], i)
        logger.add_scalar('train/trans_cost', trans_cost / 1e6)
        logger.add_scalar('train/lr', cur_lr, i)

        if i % args.test_freq == 0:
            test_monitor = VariableMonitor(['acc', 'loss'])
            for j in range(args.client.client_num):
                test_monitor.append(group.clients[j].test())
            mean_test_variables = test_monitor.variable_mean()
            for k in mean_test_variables:
                logger.add_scalar('test_after_aggregation/{}'.format(k), mean_test_variables[k], i)
            torch.save(group.server['glob_dict'], os.path.join(args.logging_path, 'model.ckpt'))

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
