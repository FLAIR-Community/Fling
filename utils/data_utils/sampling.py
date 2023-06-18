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


def data_sampling(dataset, args):
    if args.data.sample_method.name == 'iid':
        return iid_sampling(dataset, args.client_num)
    elif args.data.sample_method.name == 'dirichlet':
        data_labels = np.stack([dataset[i][1] for i in range(len(dataset))], axis=0)
        indexes = dirichlet_sampling(data_labels, args.client_num, alpha=args.data.sample_method.alpha)
        return [MyDataset(tot_data=dataset, indexes=indexes[i]) for i in range(args.client_num)]
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


def dirichlet_sampling(dataset, client_number, alpha):
    n_classes = 10
    # (#label, #client) label_distribution[i, j] means the ratio of samples of class i in client j.
    # Thus, np.sum(label_distribution, axis=1) = np.ones()
    label_distribution = np.random.dirichlet([alpha] * client_number, n_classes)

    # recording the sample indexes for each class
    class_indexes = [np.argwhere(dataset == y).flatten() for y in range(n_classes)]
    for item in class_indexes:
        np.random.shuffle(item)

    client_indexes = [[] for _ in range(client_number)]
    for c, fracs in zip(class_indexes, label_distribution):
        # c: total indexes for each class
        # fracs: distribution over each client
        for i, idc in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_indexes[i] += [idc]

    client_indexes = [np.concatenate(idc) for idc in client_indexes]

    return client_indexes
