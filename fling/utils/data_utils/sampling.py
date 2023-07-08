import random
import numpy as np

from torch.utils import data


class MyDataset(data.Dataset):

    def __init__(self, tot_data, indexes):
        self.tot_data = tot_data
        self.indexes = indexes

    def __getitem__(self, item):
        return self.tot_data[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)


def data_sampling(dataset, args, seed, train=True):
    sample_num = args.data.sample_method.train_num if train else args.data.sample_method.test_num
    if args.data.sample_method.name == 'iid':
        return iid_sampling(dataset, args.client.client_num)
    elif args.data.sample_method.name == 'dirichlet':
        indexes = dirichlet_noniid(dataset.dataset, args.client.client_num, args.data.sample_method.alpha, sample_num, seed)
        return [MyDataset(tot_data=dataset, indexes=indexes[i]) for i in range(args.client.client_num)]
    elif args.data.sample_method.name == 'pathological':
        indexes = pathological_noniid(dataset.dataset, args.client.client_num, args.data.sample_method.alpha, sample_num, seed)
        return [MyDataset(tot_data=dataset, indexes=indexes[i]) for i in range(args.client.client_num)]
    else:
        raise ValueError(f'Unrecognized sampling method: {args.data.sample_method.name}')


def iid_sampling(dataset, client_number):
    num_items = int(len(dataset) / client_number)
    dict_users, all_index = {}, [i for i in range(len(dataset))]
    for i in range(client_number - 1):
        dict_users[i] = random.sample(all_index, num_items)
        all_index = list(set(all_index).difference(set(dict_users[i])))
    dict_users[client_number - 1] = all_index

    return [MyDataset(tot_data=dataset, indexes=dict_users[i]) for i in range(len(dict_users))]


def pathological_noniid(dataset, num_users, alpha, sample_num, seed):
    num_indices = len(dataset.targets)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # if sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // num_users

    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(num_users)]
    random_state = np.random.RandomState(seed)

    class_idxs = [i for i in range(num_classes)]
    for i in range(num_users):
        class_idx = random_state.choice(class_idxs, alpha, replace=False)
        for j in class_idx:
            selected = random_state.choice(idxs_classes[j], int(sample_num / alpha), replace=False)
            client_indexes[i] += list(selected)
        client_indexes[i] = np.array(client_indexes[i])
    return client_indexes

def dirichlet_noniid(dataset, num_users, alpha, sample_num, seed):
    num_indices = len(dataset.targets)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # if sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // num_users

    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(num_users)]
    random_state = np.random.RandomState(seed)
    q = random_state.dirichlet(np.repeat(alpha, num_classes), num_users)

    for i in range(num_users):
        # make sure that each client have sample_num samples
        temp_sample = sample_num

        # partition each class for clients
        for j in range(num_classes):
            select_num = int(sample_num * q[i][j] + 0.5)
            select_num = select_num if temp_sample - select_num >= 0 else temp_sample
            temp_sample -= select_num
            assert select_num >= 0
            selected = random_state.choice(idxs_classes[j], select_num, replace=False)
            client_indexes[i] += list(selected)
        client_indexes[i] = np.array(client_indexes[i])
    return client_indexes


