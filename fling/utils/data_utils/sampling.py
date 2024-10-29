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


def uniform_sampling(dataset: Dataset, client_number: int, sample_num: int, seed: int) -> List:
    r"""
    Overview:
        Uniform sampling method. For each client, the dataset is sampled randomly from the total dataset. \
    The datasets used for different clients will not overlap under this setting. This may be slightly from standard \
    "iid scenario". For concrete differences, please refer to ``iid_sampling``.
    Arguments:
        dataset: The total dataset to be sampled from
        client_number: The number of clients
        sample_num: The number of samples in each client. If the value is zero, the number will be
            ``len(dataset) // client_number``.
        seed: Dynamic seed.
    Returns:
        A list of datasets for each client.
    """
    num_indices = len(dataset)

    # If sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    assert sample_num * client_number <= num_indices, "Total data required is larger than original dataset. " \
                                                      "It is not permitted in ``uniform_sampling``. " \
                                                      "Please try ``iid_sampling`` instead."

    random_state = np.random.RandomState(seed)

    dict_users, all_index = {}, [i for i in range(len(dataset))]

    random_state.shuffle(all_index)
    length_per_client = num_indices // client_number
    for i in range(client_number):
        start_idx = i * length_per_client
        dict_users[i] = all_index[start_idx:start_idx + sample_num]

    return [NaiveDataset(tot_data=dataset, indexes=dict_users[i]) for i in range(len(dict_users))]


def iid_sampling(dataset: Dataset, client_number: int, sample_num: int, seed: int) -> List:
    r"""
    Overview:
        Independent and identical (i.i.d) sampling method. For each client, the dataset is sampled randomly from the \
    total dataset independently. It should be noted that, datasets for different clients may have same data samples, \
    which is different from ``uniform_sampling``.
    Arguments:
        dataset: The total dataset to be sampled from
        client_number: The number of clients
        sample_num: The number of samples in each client. If the value is zero, the number will be
            ``len(dataset) // client_number``.
        seed: Dynamic seed.
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


def pathological_sampling(dataset: Dataset, client_number: int, sample_num: int, seed: int, alpha: int) -> List:
    r"""
    Overview:
        Pathological sampling method. The overall steps are as follows:
        1) Randomly generate the data distribution for each client, i.e. decide which data classes do each client have.
        2) Sample the total dataset to satisfy the distribution required by each client.
        Note: The data samples for each client may overlap.
    Arguments:
        dataset: The total dataset to be sampled from
        client_number: The number of clients
        sample_num: the number of samples in each client. If the value is zero, the number will be
         ``len(dataset) // client_number``.
        seed: Dynamic seed.
        alpha: How many classes in each client.
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
        Dirichlet sampling method. The overall steps are as follows:
        1) Randomly generate the data distribution for each client, i.e. decide the number of samples for each class \
         in each client. The generation of such distributions obeys dirichlet distribution.
        2) Sample the total dataset to satisfy the distribution required by each client.
        Note: The data samples for each client may overlap.
    Arguments:
        dataset: The total dataset to be sampled from
        client_number: The number of clients
        sample_num: The number of samples in each client. If the value is zero, the number will be
         ``len(dataset) // client_number``.
        seed: Dynamic seed.
        alpha: The argument alpha in dirichlet sampling with range (0, +inf).
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
    'uniform': uniform_sampling,
    'iid': iid_sampling,
    'dirichlet': dirichlet_sampling,
    'pathological': pathological_sampling,
}


def data_sampling(dataset: Dataset, args: dict, seed: int, train: bool = True) -> List:
    r"""
    Overview:
        Sample the data set using the given configurations.
    Arguments:
        dataset: The total dataset to be sampled from.
        args: Arguments of the current experiment.
        seed: Dynamic seed for better reproducibility.
        train: Whether this sampling is for training dataset or testing dataset.
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
 
    # For Cross-Domain Scenario: clients in the same domain share an identical test dataset.
    if 'domains' in args['data'] and train is not True:
        return sampling_func(dataset, 1, len(dataset), seed, **sampling_config)
    else:
        return sampling_func(dataset, args.client.client_num, sample_num, seed, **sampling_config)
