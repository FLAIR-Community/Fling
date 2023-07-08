from copy import deepcopy
import numpy as np
from torch.utils import data


class NaiveDataset(data.Dataset):

    def __init__(self, tot_data, indexes):
        self.tot_data = tot_data
        self.indexes = indexes

    def __getitem__(self, item):
        return self.tot_data[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)


def iid_sampling(dataset, client_number, sample_num, seed):
    # if sample_num is not specified, then the dataset is divided equally among each client
    num_indices = len(dataset)
    if sample_num == 0:
        sample_num = num_indices // client_number

    random_state = np.random.RandomState(seed)

    dict_users, all_index = {}, [i for i in range(len(dataset))]
    for i in range(client_number - 1):
        dict_users[i] = random_state.choice(all_index[j], sample_num, replace=False)
    dict_users[client_number - 1] = all_index

    return [NaiveDataset(tot_data=dataset, indexes=dict_users[i]) for i in range(len(dict_users))]


def pathological_sampling(dataset, client_number, sample_num, seed, alpha):
    num_indices = len(dataset.targets)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # if sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(client_number)]
    random_state = np.random.RandomState(seed)

    class_idxs = [i for i in range(num_classes)]
    for i in range(client_number):
        class_idx = random_state.choice(class_idxs, alpha, replace=False)
        for j in class_idx:
            selected = random_state.choice(idxs_classes[j], int(sample_num / alpha), replace=False)
            client_indexes[i] += list(selected)
        client_indexes[i] = np.array(client_indexes[i])
    return [NaiveDataset(tot_data=dataset, indexes=client_indexes[i]) for i in range(client_number)]


def dirichlet_sampling(dataset, client_number, sample_num, seed, alpha):
    num_indices = len(dataset.targets)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # if sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(client_number)]
    random_state = np.random.RandomState(seed)
    q = random_state.dirichlet(np.repeat(alpha, num_classes), client_number)

    for i in range(client_number):
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
    return [NaiveDataset(tot_data=dataset, indexes=client_indexes[i]) for i in range(client_number)]


sampling_methods = {
    'iid': iid_sampling,
    'dirichlet': dirichlet_sampling,
    'pathological': pathological_sampling,
}


def data_sampling(dataset, args, seed, train=True):
    sampling_config = deepcopy(args.data.sample_method)
    train_num, test_num = sampling_config.pop('train_num'), sampling_config.pop('test_num')
    sample_num = train_num if train else test_num
    sampling_name = sampling_config.pop('name')
    try:
        sampling_func = sampling_methods[sampling_name]
    except KeyError:
        raise ValueError(f'Unrecognized sampling method: {args.data.sample_method.name}')
    return sampling_func(dataset, args.client.client_num, sample_num, seed, **sampling_config)
