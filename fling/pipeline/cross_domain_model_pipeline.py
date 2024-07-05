import os
import torch

from fling.component.client import get_cross_domain_client
from fling.component.server import get_server
from fling.component.group import get_cross_domain_group
from fling.dataset import get_cross_domain_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, VariableMonitor, LRScheduler, get_cross_domain_launcher


def cross_domain_model_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for personalized federated learning. Under this setting, models of each client is different.
       The final performance of is calculated by averaging the local model in each client.
       Typically, each local model is tested using local test dataset.
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed=seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    
    # Load dataset with domains.
    # edition: train_set[domain]
    domains = args.data.domains.split(',')
    train_set = {}
    test_set = {}
    
    for domain in domains:
        train_set[domain] = get_cross_domain_dataset(args, domain, train=True)
        test_set[domain] = get_cross_domain_dataset(args, domain, train=False)
    

    # Split dataset into clients.
    # edition: train_sets_loader[domain][user id]
    train_sets, train_sets_len = data_sampling(train_set, args, seed, train=True)
    test_sets, test_sets_len = data_sampling(train_set, args, seed, train=False)


    # Initialize group, clients and server.
    # edition: new pipeline client, server for cross-domain
    group = get_cross_domain_group(args, logger)
    group.server = get_server(args, test_dataset=test_sets)
    # edition:according to num_user per domain to create client
    for domain in domains:
        for i in range(args.client.client_num):
            group.append(domain,
                get_cross_domain_client(
                    args=args,
                    domain=domain,
                    client_id=i,
                    train_dataset=train_sets[domain][i],
                    test_dataset=test_sets[domain][0]
                )
            )
    group.initialize()
    # ********************** to be edited *************************

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(args)

    # Setup launcher.
    launcher = get_cross_domain_launcher(args)

    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        # Initialize variable monitor.
        train_monitor = VariableMonitor()

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)

        # Local training for each participated client and add results to the monitor.
        # Use multiprocessing for acceleration.
        # edition: all clients from all domains will be participated in each communication round.
        train_results = launcher.launch(
            clients=[group.clients[domain][client] for domain in domains for client in range(args.client.client_num)]
                                        , lr=cur_lr, task_name='train')
        
        for item in train_results:
            train_monitor.append(item)

        # Testing
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()

            # Testing for each client and add results to the monitor
            # Use multiprocessing for acceleration.
            test_results = launcher.launch(
                clients=[group.clients[domain][client] for domain in domains for client in range(args.client.client_num)]
                                           , task_name='test')
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
                clients=[group.clients[domain][client] for domain in domains for client in range(args.client.client_num)]
                                           , task_name='test')
            for item in test_results:
                test_monitor.append(item)

            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=mean_test_variables, rnd=i)

            # Saving model checkpoints.
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))

    # # Fine-tuning
    # # Fine-tune model on each client and collect all the results.
    # finetune_results = launcher.launch(
    #     clients=[group.clients[j] for j in range(client_num)],
    #     lr=cur_lr,
    #     finetune_args=args.learn.finetune_parameters,
    #     task_name='finetune'
    # )

    # # Logging fine-tune results
    # for key in finetune_results[0][0].keys():
    #     for eid in range(len(finetune_results[0])):
    #         tmp_mean = sum([finetune_results[cid][eid][key]
    #                         for cid in range(len(finetune_results))]) / len(finetune_results)
    #         logger.add_scalar(f'finetune/{key}', tmp_mean, eid)
