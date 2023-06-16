import os
import tqdm
import numpy as np

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from component.client import Client
from component.server import Server
from component.group import ParameterServerGroup
from utils.data_utils import DatasetConstructor, sample
from utils import Logger, get_params_number, seed_everything


def personalized_model_serial_pipeline(args, seed=0):
    seed_everything(seed)
    client_pool = ParameterServerGroup(args)
    tb_logger = SummaryWriter(args.logging_path)
    logger = Logger(args.logging_path)

    # load dataset
    train_set = DatasetConstructor(args).get_dataset()
    test_set = DatasetConstructor(args).get_dataset(train=False)
    if 'dataloader_type' in args.__dict__.keys() and args.dataloader_type == 'nlp':
        test_set = sample(
            args.sample_method, test_set, args.client_num, alpha=args.alpha, args=args
        )[0]

    # split dataset into clients. alpha affects the distribution for dirichlet non-iid sampling.
    # If you don't use dirichlet, this parameter can be omitted.
    train_sets = sample(
        args.sample_method, train_set, args.client_num, alpha=args.alpha, args=args
    )

    # if you need all clients to test locally use next line to split test sets
    # test_sets = sample(args.sample_method, test_set, args.client_num)

    # initialize clients, assemble datasets
    for i in range(args.client_num):
        client_pool.append(Client(train_sets[i], args=args, client_id=i, test_dataset=None, test_frac=0.2))
    print(client_pool[0].model)
    logger.logging('All clients initialized.')
    logger.logging('Parameter number in each model: {:.2f}M'.format(get_params_number(client_pool[0].model) / 1e6))

    # global initialization
    if args.fed_dict == 'all':
        glob_dict = client_pool[0].model.state_dict()
    elif args.fed_dict == 'except_bn':
        state_dict = client_pool[0].model.state_dict()
        glob_dict = {}
        for key in state_dict:
            if 'downsample.1' not in key and 'bn' not in key:
                glob_dict[key] = state_dict[key]
    elif args.fed_dict == 'no_side':
        state_dict = client_pool[0].model.state_dict()
        glob_dict = {}
        for key in state_dict:
            if 'side_conv' not in key:
                glob_dict[key] = state_dict[key]
    elif args.fed_dict == 'no_fc':
        state_dict = client_pool[0].model.state_dict()
        glob_dict = {}
        for key in state_dict:
            if 'fc' not in key:
                glob_dict[key] = state_dict[key]
    else:
        glob_dict = client_pool[0].get_state_dict(args.fed_dict)
    if args.resume:
        glob_dict = torch.load('./model_checkpoints/model.ckpt')

    dataloader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    server = Server(args, args.device, dataloader)
    server.add_attr(name='glob_dict', item=glob_dict)
    client_pool.server = server

    # set fed keys in each client and init compression settings
    client_pool.set_fed_keys()
    client_pool.sync()
    train_accuracies = []
    train_losses = []
    trans_costs = []

    # training loop
    for i in range(args.start_round, args.glob_eps):
        # A communication begins.
        train_acc = 0
        train_loss = 0
        total_client = 0
        print('Starting round: ' + str(i))

        # Random sample participated clients in each communication round.
        participated_clients = np.array(range(args.client_num))
        participated_clients = sorted(
            list(
                np.random.choice(
                    participated_clients,
                    int(args.client_sample_rate * participated_clients.shape[0]),
                    replace=False
                )
            )
        )

        # Adjust learning rate.
        cur_lr = args.lr * (args.decay_factor ** i)
        for j in tqdm.tqdm(participated_clients):
            # Local training in each client.
            total_client += 1
            client = client_pool[j]
            acc, loss = client.train(
                lr=cur_lr,
                momentum=args.momentum,
                optimizer=args.optimizer,
                loss=args.loss,
                local_eps=args.loc_eps
            )

            train_acc += acc
            train_loss += loss

        # Test before aggregation.
        if i % args.test_freq == 0:
            tmp_test_acc = []
            tmp_test_loss = []
            for j in participated_clients:
                # Local training in each client.
                client = client_pool[j]
                acc, loss = client.test('CrossEntropyLoss')
                tmp_test_acc.append(acc)
                tmp_test_loss.append(loss)
            tmp_test_acc = sum(tmp_test_acc) / len(tmp_test_acc)
            tmp_test_loss = sum(tmp_test_loss) / len(tmp_test_loss)
            tb_logger.add_scalar('before_test/acc', tmp_test_acc, i)
            tb_logger.add_scalar('before_test/loss', tmp_test_loss, i)

        # Aggregation and sync.
        trans_cost = client_pool.aggregate(i, tb_logger=tb_logger)

        # Logging
        train_accuracies.append(train_acc / total_client)
        train_losses.append(train_loss / total_client)
        trans_costs.append(trans_cost)
        logger.logging(
            'epoch:{}, train_acc: {:.4f}, train_loss: {:.4f}, trans_cost: {:.4f}M'.format(
                i, train_accuracies[-1], train_losses[-1], trans_costs[-1] / 1e6
            )
        )
        tb_logger.add_scalar('train/acc', train_accuracies[-1], i)
        tb_logger.add_scalar('train/loss', train_losses[-1], i)
        tb_logger.add_scalar('train/lr', cur_lr, i)

        # Test after aggregation.
        if i % args.test_freq == 0:
            tmp_test_acc = []
            tmp_test_loss = []
            for j in participated_clients:
                # Local training in each client.
                client = client_pool[j]
                acc, loss = client.test('CrossEntropyLoss')
                tmp_test_acc.append(acc)
                tmp_test_loss.append(loss)
            tmp_test_acc = sum(tmp_test_acc) / len(tmp_test_acc)
            tmp_test_loss = sum(tmp_test_loss) / len(tmp_test_loss)
            tb_logger.add_scalar('after_test/acc', tmp_test_acc, i)
            tb_logger.add_scalar('after_test/loss', tmp_test_loss, i)

            logger.logging('epoch:{}, test_acc: {:.4f}, test_loss: {:.4f}'.format(i, tmp_test_acc, tmp_test_loss))

            if not os.path.exists('./model_checkpoints'):
                os.makedirs('./model_checkpoints')
            torch.save(client_pool.server['glob_dict'], './model_checkpoints/model.ckpt')
    #################################################
    # Finetune and calculate final accuracy & loss. #
    #################################################
    tmp_test_acc = []
    tmp_test_loss = []
    tmp_train_acc = []
    tmp_train_loss = []
    for j in tqdm.tqdm(range(len(client_pool.clients))):
        # Local training in each client.
        client = client_pool[j]
        # Get train_acc, train_loss, test_acc, test_loss
        tmp_dict = client.finetune(
            lr=args.lr,
            momentum=args.momentum,
            optimizer=args.optimizer,
            loss_name=args.loss,
            local_eps=args.loc_eps
        )
        acc, loss, tmp_ta, tmp_tl = tmp_dict['train_acc'], tmp_dict['train_loss'], \
                                    tmp_dict['test_acc'], tmp_dict['test_loss']
        tmp_train_acc.append(acc)
        tmp_train_loss.append(loss)
        tmp_test_acc.append(tmp_ta)
        tmp_test_loss.append(tmp_tl)
    # Logging the relevant metrics.
    methods_keys = tmp_test_acc[0].keys()
    for kk in methods_keys:
        tta = [c[kk] for c in tmp_train_acc]
        ttl = [c[kk] for c in tmp_train_loss]

        tta = sum(tta) / len(tta)
        ttl = sum(ttl) / len(ttl)
        tb_logger.add_scalar(f'finetune/{kk}_train_acc', tta, 0)
        tb_logger.add_scalar(f'finetune/{kk}_train_loss', ttl, 0)

        tta = [c[kk] for c in tmp_test_acc]
        ttl = [c[kk] for c in tmp_test_loss]

        for i in range(len(tta[0])):
            mean_acc = sum([tta[j][i] for j in range(len(tta))]) \
                       / len([tta[j][i] for j in range(len(tta))])
            mean_loss = sum([ttl[j][i] for j in range(len(ttl))]) \
                        / len([ttl[j][i] for j in range(len(ttl))])
            tb_logger.add_scalar(f'finetune/{kk}_test_acc', mean_acc, i)
            tb_logger.add_scalar(f'finetune/{kk}_test_loss', mean_loss, i)


