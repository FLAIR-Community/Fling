import copy
import os
import torch
from torch.utils.data import DataLoader

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset

from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler
from fling.utils import get_launcher
from fling.utils import plot_2d_loss_landscape


def generic_model_visualization_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for generic federated learning. Under this setting, models of each client is the same.
       We plot the loss landscape before and after aggregation in each round.
       The final performance of this generic model is tested on the server (typically using a global test dataset).
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset.
    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)

    part_test = [test_set[i] for i in range(100)]
    part_test = DataLoader(part_test, batch_size=args.learn.batch_size, shuffle=True)

    # Split dataset into clients.
    train_sets = data_sampling(train_set, args, seed, train=True)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)
    for i in range(args.client.client_num):
        group.append(get_client(args=args, client_id=i, train_dataset=train_sets[i]))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.optimizer.lr, args=args.learn.scheduler)
    # Setup launcher.
    launcher = get_launcher(args)

    # Variables for visualization
    last_global_model = None

    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        # Initialize variable monitor.
        train_monitor = VariableMonitor()

        # Random sample participated clients in each communication round.
        participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)

        # Local training for each participated client and add results to the monitor.
        # Use multiprocessing for acceleration.
        train_results = launcher.launch(
            clients=[group.clients[j] for j in participated_clients], lr=cur_lr, task_name='train'
        )
        for item in train_results:
            train_monitor.append(item)

        # Testing
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)
            # Logging test variables.
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=test_result, rnd=i)

        if last_global_model is not None and i > 45:
            plot_2d_loss_landscape(
                model=last_global_model,
                dataloader=part_test,
                device=args.learn.device,
                caption='Global-test Loss Landscape',
                save_path=os.path.join(args.other.logging_path, f"losslandscape_gt_{i}.pdf"),
                target_model1=group.clients[0].model,
                target_model2=group.clients[1].model,
                resolution=20,
                noise_range=(-0.1, 1.0),
                log_scale=True,
                max_val=20
            )
            plot_2d_loss_landscape(
                model=last_global_model,
                dataloader=group.clients[0].train_dataloader,
                device=args.learn.device,
                caption='Client-1-train Loss Landscape',
                save_path=os.path.join(args.other.logging_path, f"losslandscape_ct1_{i}.pdf"),
                target_model1=group.clients[0].model,
                target_model2=group.clients[1].model,
                resolution=20,
                noise_range=(-0.1, 1.1),
                log_scale=True,
                max_val=20
            )
            plot_2d_loss_landscape(
                model=last_global_model,
                dataloader=group.clients[1].train_dataloader,
                device=args.learn.device,
                caption='Client-2-train Loss Landscape',
                save_path=os.path.join(args.other.logging_path, f"losslandscape_ct2_{i}.pdf"),
                target_model1=group.clients[0].model,
                target_model2=group.clients[1].model,
                resolution=20,
                noise_range=(-0.1, 1.1),
                log_scale=True,
                max_val=20
            )

        # Aggregate parameters in each client.
        trans_cost = group.aggregate(i)

        last_global_model = copy.deepcopy(group.clients[0].model)

        # Logging train variables.
        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update({'trans_cost': trans_cost / 1e6, 'lr': cur_lr})
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)

        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=test_result, rnd=i)

            # Saving model checkpoints.
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))
