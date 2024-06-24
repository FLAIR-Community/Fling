# How to Add a New Dataset

We have already support some common dataset to satisfy common usages, such as: CIFAR10, CIFAR100, MNIST, ...

If you want to add a new dataset to Fling, please refer to the following steps.

### Step 1: Add a dataset file

In this step, you are required to define a dataset in `fling/dataset` . An example is `fling/dataset/cifar100.py`, which is shown as below:

```python
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('cifar100')
class CIFAR100Dataset(Dataset):
    r"""
        Implementation for CIFAR100 dataset. Details can be viewed in: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    default_augmentation = dict(
        horizontal_flip=dict(p=0.5),
        random_rotation=dict(degree=15),
        Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        random_crop=dict(size=32, padding=4),
    )

    def __init__(self, cfg: dict, train: bool):
        super(CIFAR100Dataset, self).__init__()
        self.train = train
        self.cfg = cfg
        transform = get_data_transform(cfg.data.transforms, train=train)
        self.dataset = CIFAR100(cfg.data.data_path, train=train, transform=transform, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict:
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}
```

Notice that:

- You should give a name to the dataset by using this decorator: `@DATASET_REGISTRY.register('cifar100')`
- The dataset defined should be a subclass of `torch.utils.data.Dataset`
- The `default_augmentation` refers to the default data augmentation method of this dataset. If you do not explicitly define this attribute, no augmentation will be applied by default. More information about how to override this default setting in user-defined config, please refer to this [link](https://github.com/FLAIR-Community/Fling/blob/main/docs/meaning_for_configurations_en.md).
- For classification tasks, the return data item should in this diction format: `{'input': x, 'class_id': y} `. If you are not conducting a classification task, please define the format yourself and modify the data-preprocessing and learning operations  according step 5.

### Step 2: Import the dataset file

When you add a new dataset file, don't forget to import it in `fling.dataset.__init__.py`:

```python
from .cifar100 import CIFAR100Dataset
```

### Step 3: Prepare your configuration file

After the previous steps, you can write your configuration file now to use your own dataset !

```python
data=dict(
        dataset='cifar100',
        data_path='./data/CIFAR100',
        sample_method=dict(name='iid', train_num=500, test_num=100)
),
```

Note that:

- The key `dataset` is the name of your dataset registered in step 1
- `dataset_path` refer to the path to store your dataset
- `sample_method` refer to the criterion to sample data for each client. For classification tasks, you can use "iid", "dirichlet", "pathological", but for non-classification tasks, only "iid" is available.

After this step, if your dataset is a classification task, the whole process is finished. However, if you still need to modify the learning process, please refer to the following steps.

### Step 4: Modify data-preprocessing operations

In this step, your are required to define the data-preprocessing operations in your client. By default, this step contains two operations:

```python
def preprocess_data(self, data):
    return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}
```

- Put the data to device (CUDA or CPU).
- Pick out the input and class_id in input data. 

### Step 5: Modify learning operations

In this step, your are required to define the learning operations in your client. By default, this step is defined as below:

```python
def train_step(self, batch_data, criterion, monitor, optimizer):
    batch_x, batch_y = batch_data['x'], batch_data['y']
    # Forward calculation
    o = self.model(batch_x)
    loss = criterion(o, batch_y)
    # Predict the label
    y_pred = torch.argmax(o, dim=-1)
    # Record the acc and loss. Add the results to monitor.
    monitor.append(
        {
            'train_acc': torch.mean((y_pred == batch_y).float()).item(),
            'train_loss': loss.item()
        },
        weight=batch_y.shape[0]
    )
    # Step.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Also, you should define the corresponding testing process:

```python
def test_step(self, batch_data, criterion, monitor):
    batch_x, batch_y = batch_data['x'], batch_data['y']
    # Forward calculation
    o = self.model(batch_x)
    loss = criterion(o, batch_y)
    # Predict the label
    y_pred = torch.argmax(o, dim=-1)
    # Record the acc and loss. Add the results to monitor.
    monitor.append(
        {
            'test_acc': torch.mean((y_pred == batch_y).float()).item(),
            'test_loss': loss.item()
        },
        weight=batch_y.shape[0]
    )
```

