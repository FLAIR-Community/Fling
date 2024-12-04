import os
import torch

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, VariableMonitor, LRScheduler, get_launcher


def cross_domain_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       The pipeline is designed for federated learning in cross-domain scenarios. 
       In this setup, the data on clients originates from different domains.
       In the cross-domain pipeline, the hyperparameter `client_num` indicates the number of clients originating from the same domain. 
       The final performance is evaluated through local testing on clients.
       Importantly, clients within the same domain share the same test data.

    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed=seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset with domains.
    domains = args.data.domains.split(',')
    train_set = {}
    test_set = {}
    for domain in domains:
        train_set[domain] = get_dataset(args, train=True, domain=domain)
        test_set[domain] = get_dataset(args, train=False, domain=domain)

    # Split dataset into clients.
    train_sets = {}
    test_sets = {}
    for domain in domains:
        train_sets[domain] = data_sampling(train_set[domain], args, seed, train=True)
        test_sets[domain] = data_sampling(test_set[domain], args, seed, train=False)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_sets)
    for domain in domains:
        for i in range(args.client.client_num):
            group.append(
                domain,
                get_client(
                    args=args,
                    client_id=i,
                    train_dataset=train_sets[domain][i],
                    test_dataset=test_sets[domain][0],
                    domain=domain
                )
            )
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.optimizer.lr, args=args.learn.scheduler)

    # Setup launcher.
    launcher = get_launcher(args)

    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        # Initialize variable monitor.
        train_monitor = VariableMonitor()

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)

        # Local training for each participated client and add results to the monitor.
        # Use multiprocessing for acceleration.
        train_results = launcher.launch(
            clients=[group.clients[domain][client] for domain in domains for client in range(args.client.client_num)],
            lr=cur_lr,
            task_name='train'
        )

        for item in train_results:
            train_monitor.append(item)

        # Testing
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()

            # Testing for each client and add results to the monitor
            # Use multiprocessing for acceleration.
            test_results = launcher.launch(
                clients=[
                    group.clients[domain][client] for domain in domains for client in range(args.client.client_num)
                ],
                task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)

            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()

            # Logging test variables.
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=mean_test_variables, rnd=i)

        # Aggregate parameters in each client.
        trans_cost = group.aggregate(i)

        # Logging for train variables.
        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update({'trans_cost(MB)': trans_cost / 1e6, 'lr': cur_lr})
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)

        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()

            # Testing for each client and add results to the monitor
            # Use multiprocessing for acceleration.
            test_results = launcher.launch(
                clients=[
                    group.clients[domain][client] for domain in domains for client in range(args.client.client_num)
                ],
                task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)

            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=mean_test_variables, rnd=i)

            # Saving model checkpoints.
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))
