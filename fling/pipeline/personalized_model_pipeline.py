import os
import tqdm
import torch

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler


def personalized_model_serial_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for pesonalized federated learning. Under this setting, models of each client is different.
       The final performance of is calculated by averaging the local model in each client.
       Typically each local model is tested using local test dataset.
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    Returns:
        - None
    """
    # Compile the input arguments first.
    args = compile_config(args, seed=seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset.
    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)
    # Split dataset into clients.
    train_sets = data_sampling(train_set, args)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)
    for i in range(args.client.client_num):
        group.append(get_client(train_sets[i], args=args, client_id=i))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(args)

    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        # Initialize variable monitor.
        train_monitor = VariableMonitor(['train_acc', 'train_loss'])

        # Random sample participated clients in each communication round.
        participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)

        # Local training for each participated client and add results to the monitor.
        for j in tqdm.tqdm(participated_clients):
            train_monitor.append(group.clients[j].train(lr=cur_lr))

        # Aggregate parameters in each client.
        trans_cost = group.aggregate(i)

        # Logging for train variables.
        mean_train_variables = train_monitor.variable_mean()
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)
        extra_info = {'trans_cost': trans_cost / 1e6, 'lr': cur_lr}
        logger.add_scalars_dict(prefix='train', dic=extra_info, rnd=i)

        # Testing
        if i % args.other.test_freq == 0:
            test_monitor = VariableMonitor(['test_acc', 'test_loss'])

            # Testing for each client and add results to the monitor
            for j in range(args.client.client_num):
                test_monitor.append(group.clients[j].test())
            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()

            # Logging test variables.
            logger.add_scalars_dict(prefix='test', dic=mean_test_variables, rnd=i)

            # Saving model checkpoints.
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))

    # Fine-tuning
    # Fine-tune model on each client and collect all the results.
    finetune_results = [
        group.clients[i].finetune(lr=cur_lr, finetune_args=args.learn.finetune_parameters)
        for i in range(len(group.clients))
    ]

    # Logging fine-tune results
    for key in finetune_results[0][0].keys():
        for eid in range(len(finetune_results[0])):
            tmp_mean = sum([finetune_results[cid][eid][key]
                            for cid in range(len(finetune_results))]) / len(finetune_results)
            logger.add_scalar(f'finetune/{key}', tmp_mean, eid)
