# 如何添加新数据集

我们已支持一些常见数据集以满足普遍的使用，例如：CIFAR10、CIFAR100、MNIST 等。

如果您想向 Fling 添加新的数据集，请参考以下步骤：

### 步骤 1：添加数据集文件

在这一步中，您需要在 `fling/dataset` 中定义一个数据集。以 `fling/dataset/cifar100.py` 为例，如下所示：

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

请注意：

- 您应该使用此注册器为数据集命名：`@DATASET_REGISTRY.register('cifar100')`
- 您定义的数据集应为 `torch.utils.data.Dataset` 的子类。
- `default_augmentation` 指的是该数据集默认的数据增强方法。如果您没有显式定义这个属性，默认情况下不会使用任何增强。关于如何在用户定义的配置中覆盖此默认配置的更多信息，请参阅[此链接](https://github.com/FLAIR-Community/Fling/blob/main/docs/meaning_for_configurations_en.md)。
- 对于分类任务，返回的数据项应该具有以下字典格式：`{'input': x, 'class_id': y}`。如果您进行的不是分类任务，请自行定义格式，并根据 **步骤 5** 修改数据预处理和学习操作。

### 步骤 2：导入数据集文件

当您添加一个新的数据集文件时，别忘了在 `fling.dataset.__init__.py` 中导入它：

```python
from .cifar100 import CIFAR100Dataset
```

### 步骤 3：准备您的配置文件

在完成前面的步骤之后，您现在可以编写配置文件，以使用您自己的数据集了！

```python
data=dict(
        dataset='cifar100',
        data_path='./data/CIFAR100',
        sample_method=dict(name='iid', train_num=500, test_num=100)
),
```

请注意：

- `dataset` 的键值是您在 **步骤 1** 中注册的数据集的名称。
- `dataset_path` 指的是用于存储您的数据集的路径。
- `sample_method` 指的是为每个客户端抽样数据的标准。对于分类任务，您可以使用 "iid"、"dirichlet"、"pathological"，但对于非分类任务，只有 "iid" 可用。

在完成这一步后，如果您的数据集是一个分类任务，整个流程就已经完成了。但如果您仍需要修改学习过程，请参考以下步骤。

### 步骤 4：修改数据预处理操作

在这一步中，您需要在您的客户端中定义数据预处理操作。默认情况下，这一步包含两个操作：

```python
def preprocess_data(self, data):
    return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}
```

- 将数据置于相应设备上（CUDA或CPU）。
- 从输入数据中提取出输入和类别ID（class_id）。

### 步骤 5：修改学习操作

在这一步中，您需要在您的客户端中定义学习操作。默认情况下，这一步的定义如下所示：

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

同样地，您也应该定义对应的测试过程：

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

