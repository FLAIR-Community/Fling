from copy import deepcopy
from typing import List
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils import data


class NaiveDataset(data.Dataset):

    def __init__(self, tot_data: Dataset, indexes: List):
        self.tot_data = tot_data
        self.indexes = indexes

    def __getitem__(self, item: int) -> object:
        return self.tot_data[self.indexes[item]]

    def __len__(self) -> int:
        return len(self.indexes)


def iid_sampling(dataset: Dataset, client_number: int, sample_num: int, seed: int) -> List:
    r"""
    Overview:
        Independent and identical (i.i.d) sampling method.
    Arguments:
        dataset: the total dataset to be sampled from
        client_number: the number of clients
        sample_num: the number of samples in each client. If the value is zero, the number will be
            ``len(dataset) // client_number``.
        seed: dynamic seed.
    Returns:
        A list of datasets for each client.
    """
    num_indices = len(dataset)

    # If sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    random_state = np.random.RandomState(seed)

    dict_users, all_index = {}, [i for i in range(len(dataset))]
    for i in range(client_number):
        dict_users[i] = random_state.choice(all_index, sample_num, replace=False)

    return [NaiveDataset(tot_data=dataset, indexes=dict_users[i]) for i in range(len(dict_users))]


def cross_domain_iid_sampling(dataset: Dataset, num_users: int) -> List:
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:number of users per domain
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return [NaiveDataset(tot_data=dataset, indexes=dict_users[i]) for i in range(len(dict_users))]


def pathological_sampling(dataset: Dataset, client_number: int, sample_num: int, seed: int, alpha: int) -> List:
    r"""
    Overview:
        Pathological sampling method.
    Arguments:
        dataset: the total dataset to be sampled from
        client_number: the number of clients
        sample_num: the number of samples in each client. If the value is zero, the number will be
         ``len(dataset) // client_number``.
        seed: dynamic seed.
        alpha: how many classes in each client.
    Returns:
        A list of datasets for each client.
    """
    num_indices = len(dataset)
    labels = np.array([dataset[i]['class_id'] for i in range(num_indices)])
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # If sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    # Get samples for each class.
    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(client_number)]
    random_state = np.random.RandomState(seed)
    class_idxs = [i for i in range(num_classes)]

    # Sampling label distribution for each client
    client_class_idxs = [random_state.choice(class_idxs, alpha, replace=False) for _ in range(client_number)]

    for i in range(client_number):
        class_idx = client_class_idxs[i]
        for j in class_idx:
            # Calculate number of samples for each class.
            select_num = int(sample_num / alpha)
            # Sample a required number of samples.
            # If the number of samples in ``idx_classes[j]`` is more or equal than the required number,
            # set the argument ``replace=False``. Otherwise, set ``replace=True``
            selected = random_state.choice(idxs_classes[j], select_num, replace=(select_num > len(idxs_classes[j])))
            client_indexes[i] += list(selected)
        client_indexes[i] = np.array(client_indexes[i])
    return [NaiveDataset(tot_data=dataset, indexes=client_indexes[i]) for i in range(client_number)]


def dirichlet_sampling(dataset: Dataset, client_number: int, sample_num: int, seed: int, alpha: float) -> List:
    r"""
    Overview:
        Dirichlet sampling method.
    Arguments:
        dataset: the total dataset to be sampled from
        client_number: the number of clients
        sample_num: the number of samples in each client. If the value is zero, the number will be
         ``len(dataset) // client_number``.
        seed: dynamic seed.
        alpha: the argument alpha in dirichlet sampling with range (0, +inf).
        A smaller alpha means the distributions sampled are more imbalanced.
    Returns:
        A list of datasets for each client.
    """
    num_indices = len(dataset)
    labels = np.array([dataset[i]['class_id'] for i in range(num_indices)])
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # If ``sample_num`` is not specified, the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    # Get samples for each class.
    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(client_number)]
    random_state = np.random.RandomState(seed)
    q = random_state.dirichlet(np.repeat(alpha, num_classes), client_number)

    for i in range(client_number):
        num_samples_of_client = 0
        # Partition class-wise samples.
        for j in range(num_classes):
            # Make sure that each client have exactly ``sample_num`` samples.
            # For the last class, the number of samples is exactly the remaining sample number.
            select_num = int(sample_num * q[i][j] + 0.5) if j < num_classes - 1 else sample_num - num_samples_of_client
            select_num = min(select_num, sample_num - num_samples_of_client)
            select_num = max(select_num, 0)
            # Record current sampled number.
            num_samples_of_client += select_num
            # Sample a required number of samples.
            # If the number of samples in ``idx_classes[j]`` is more or equal than the required number,
            # set the argument ``replace=False``. Otherwise, set ``replace=True``
            selected = random_state.choice(idxs_classes[j], select_num, replace=(select_num > len(idxs_classes[j])))
            client_indexes[i] += list(selected)
        client_indexes[i] = np.array(client_indexes[i])
    return [NaiveDataset(tot_data=dataset, indexes=client_indexes[i]) for i in range(client_number)]


# Supported sampling methods.
sampling_methods = {
    'iid': iid_sampling,
    'cross_domain_iid': cross_domain_iid_sampling,
    'dirichlet': dirichlet_sampling,
    'pathological': pathological_sampling,
}


def data_sampling(dataset: Dataset, args: dict, seed: int, train: bool = True) -> List:
    r"""
    Overview:
        Dirichlet sampling method.
    Arguments:
        dataset: the total dataset to be sampled from.
        args: arguments.
        seed: dynamic seed.
        train: whether this sampling is for training dataset or testing dataset.
    Returns:
        A list of datasets for each client.
    """
    # Copy the config, or it will be modified by ``pop()``
    sampling_config = deepcopy(args.data.sample_method)
    # Determine the number of samples in each client.
    train_num, test_num = sampling_config.pop('train_num'), sampling_config.pop('test_num')
    sample_num = train_num if train else test_num
    # Determine the name of sampling methods.
    sampling_name = sampling_config.pop('name')

    # Sampling
    try:
        sampling_func = sampling_methods[sampling_name]
    except KeyError:
        raise ValueError(f'Unrecognized sampling method: {args.data.sample_method.name}')

    if sampling_name == 'cross_domain_iid':
        return sampling_func(dataset, args.client.num_users, seed, **sampling_config)
    else:
        return sampling_func(dataset, args.client.client_num, sample_num, seed, **sampling_config)
