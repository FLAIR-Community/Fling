# 配置文件各字段含义

## 默认配置实例

在本文档中，我们提供了一个完整的配置文件，并解释了每个键的含义。

请注意，以下配置文件即为默认配置文件。换句话说，如果用户定义的配置中某些键不存在，将默认使用以下文件中的值。

以下是默认配置文件，您也可以在[这里](https://github.com/FLAIR-Community/Fling/blob/main/argzoo/default_config.py)查看：

```python
import platform

default_exp_args = dict(
    # 数据集的相关配置
    data=dict(
        # 数据集的名称，必须在`DATASET_REGISTRY`中注册。
        dataset='cifar10',
        # 数据集的根路径。
        data_path='./data',
        # 图像转换方法，如：随机调整大小剪裁（RRC），尺寸调整，颜色调整...
        # 键``include_default=True``表示应用默认的数据增强方法。
        transforms=dict(include_default=True),
        # 数据集在所有客户端中的分布方式。
        sample_method=dict(
            # 'iid' 表示独立同分布（Independent and Identically Distributed）。
            name='iid',
            # 每个客户端的默认训练样本数量为500。
            train_num=500,
            # 每个客户端的默认测试样本数量为100。
            test_num=100
        )
    ),
    # 学习过程的相关配置
    learn=dict(
        # 运行深度学习模型的设备。如果只有CPU可用，将此键设置为"cpu"。
        device='0',
        # 每个客户端每轮训练的本地周期数。
        local_eps=8,
        # 整体联邦学习过程中的全局周期数（训练轮数）。
        global_eps=40,
        # 本地训练、测试和微调的批次大小。
        batch_size=32,
        # 联邦学习进行测试的时机。可选项有：'before_aggregation'，'after_aggregation'
        test_place=['after_aggregation'],
        # 本地训练中使用的优化器。
        optimizer=dict(
            # 优化器的名称。
            name='sgd',
            # 优化器的学习率。
            lr=0.02,
            # SGD优化器的动量。
            momentum=0.9
        ),
        # 学习率调度器。即对于每个全局周期使用动态变化的学习率。
        scheduler=dict(
            # 默认为"fix"，表示每个全局周期中使用的学习率保持相同。
            name='fix'
        ),
        # 哪些参数应该进行微调。
        finetune_parameters=dict(
            # 对于默认情况，应该对所有参数进行微调。
            name='all'
        ),
    ),
    # 模型的相关配置
    model=dict(
        # 使用的模型的名称。必须在`MODEL_REGISTRY`中注册。
        name='resnet8',
        # 用于初始化相应模型的参数：
        # 输入图像的通道数。
        input_channel=3,
        # 类别数，即输出 logits 的维度。
        class_number=10,
    ),
    # 客户端的相关配置
    client=dict(
        # 使用的客户端的名称。必须在`CLIENT_REGISTRY`中注册。
        name='base_client',
        # 客户端数量。
        client_num=30,
        # 每个全局周期中参与联邦学习的客户端比例。例如，若`sample_rate=0.5`，
        # 则每个全局周期中只有一半的客户端参与联邦学习。
        sample_rate=1,
        # 测试样本在总样本中的比例。例如，若`val_frac=0.2`，这意味着
        # 有20%的总数据样本将被用作本地验证数据集，剩下80%用作训练数据集。
        val_frac=0,
    ),
    # 服务器的相关配置
    server=dict(
        # 使用的服务器的名称。必须在`SERVER_REGISTRY`中注册。
        name='base_server'
    ),
    # 群组的相关配置
    group=dict(
        # 使用的群组的名称。必须在`GROUP_REGISTRY`中注册。
        name='base_group',
        # 每个客户端中的参数进行聚合的方式。默认为"avg"，表示做简单平均。
        aggregation_method='avg',
        # 哪些客户端中的参数应该进行聚合。
        aggregation_parameters=dict(
            # 对于默认情况，应聚合所有参数。
            name='all'
        ),
        # 是否对模型的统计量参数进行全局聚合。默认为"是"，表示模型的所有参数包括统计量在内都进行全局聚合。
        include_non_param=True,
    ),
    # 启动程序的相关配置
    launcher=dict(
        # 对于最简单的启动程序，串行是较合适的选择。
        name='serial'
        # 如果您想使用多进程加速训练过程，可以使用以下设置：
        # name='multiprocessing',
        # num_proc=2
        # ``num_proc``指的是程序中运行的进程数量。
        # 对于默认设置，如果您的操作系统是Linux，将启用多进程模式。
        # 您可以自行覆写默认设置。
    ) if platform.system().lower() != 'linux' else dict(name='multiprocessing', num_proc=2),
    # 其他配置
    other=dict(
        # 测试的频率。例如，`test_freq=3`表示每3个全局周期进行一次性能测试。
        test_freq=3,
        # 本实验的日志目录。
        # 如果目录不存在，将自动创建。
        # 如果目录已经存在，一些部分可能会被覆盖，应仔细检查。
        logging_path='./logging/default_experiment',
        # 保存的模型检查点以用于从特定位置开始训练。如果设置为``None``，训练过程将从头开始。
        resume_path=None,
        # 是否在命令行中打印实验配置。
        print_config=False,
    ),
)
```

## 其他教程

### learn.scheduler

此键用于控制学习率调度器。除了 "fix" 外，`learn.scheduler` 键还具有多种类型。例如，我们可以使用余弦函数定义一个调度器：

```python
scheduler=dict(
    # Cosine learning rate scheduler.
    name='cos'
    # The `min_lr` is the lower bound of learning rate.
    min_lr=1e-6,
    # This means that the learning rate will decay progressively for 30 global epochs and finally reaches `min_lr`.
    decay_round=30
)
```

对于其他类型的调度器，请参考 `fling.utils.LRScheduler`。

### learn.finetune_parameters

这个键用于控制哪些参数应该进行微调。除了 "all" 之外，`learn.finetune_parameters` 键还具有其他值。例如，如果您只想微调参数名称中**包含**关键词 "fc" 或 "bn" 的参数，可以这样写：

```python
finetune_parameters=dict(
    # This means to only fine-tune parameters whose names have keywords listed in `keywords`.
    name='contain',
    keywords=['fc', 'bn']
)
```

相反，如果您想微调参数名称**不包含**关键字 "fc" 或 "bn" 的参数，可以这样写：

```python
finetune_parameters=dict(
    # This means to only fine-tune parameters except those whose names have keywords listed in `keywords`.
    name='except',
    keywords=['fc', 'bn']
)
```

`group.aggregation_parameters` 控制在联邦学习框架下应该聚合哪些参数，对于它的设置和上面所述的 `learn.finetune_parameters` 是类似的。

为直观表示参数 `learn.finetune_parameters` 和 `group.aggregation_parameters` 两者之间含义和用法的区别，这里以 **FedPer** 算法的配置文件 [`cifar10_fedper_resnet_config.py`](https://github.com/FLAIR-Community/Fling/blob/main/fling/dataset/cifar100.py) 为例，其中对上述两种参数的定义部分如下：

```python
exp_args = dict(
    learn=dict(
        # Only fine-tune parameters whose name contain the keyword "fc".
        finetune_parameters=dict(
            name='contain',
            keywords=['fc']),
    ),
    group=dict(
        # Only aggregate parameters whose name does not contain the keyword "fc".
        aggregation_parameters=dict(
            name='except',
            keywords=['fc'],
        ),
    )
)
```

在这样的设置下进行联邦学习时，`group.aggregation_parameters` 参数使得所有客户端共享关键字 "fc" 以外的参数，而 `learn.finetune_parameters` 参数使得每个客户端能够对属于自己的、包含关键字 "fc" 的参数进行微调。

### launcher

- 除了 "serial" 外，`launcher.name` 键还有其他模式。在除了 Linux 以外的操作系统上，默认配置 "serial" 意味着每个客户端上的所有操作都是串行执行的，这可能不够高效。我们可以使用多进程方法加速这个过程。
- 在 Linux 系统上，默认的 `launcher.name` 是 "multiprocessing"，进程数为 `num_proc=2`：

```python
launcher=dict(
    name='multiprocessing',
    num_proc=2
)
```

一些需注意的重要事项：

1) 对于 Windows 用户，由于 multiprocessing + PyTorch 在 Windows 上的支持并不好，这种多进程模式可能不稳定，会导致意料之外的错误。如果您遇到与 Windows 上的多进程相关的 bug，请考虑改用串行 launcher。
2) 由于 PyTorch 2.0 的新功能，编译后的模型目前暂不支持多进程执行。在我们的设计中，如果您启用了多进程模式，PyTorch 的编译将被禁用。这个 bug 可能会在此 pr 完成后由 PyTorch 解决：https://github.com/pytorch/pytorch/pull/101651
3) `num_proc` 指的是程序中使用的进程数。这个值的最优值可能会受到您执行的任务、所使用的硬件、神经网络、数据等的影响...... 您应该在您的系统上仔细调整这个值，以寻找最佳的选择。以下是在具有 4 核 CPU 的 A100 上，batch_size=32 的简单基准测试：

| num_proc          | 1     | 2     | 3     | 4     | 8     |
| ----------------- | ----- | ----- | ----- | ----- | ----- |
| MNIST + CNN       | 21.7s | 14.9s | 15.3s | 16.5s | 25.2s |
| CIFAR10 + ResNet8 | 45.2s | 32.1s | 30.1s | 32.5s | 40.4s |

### data.transforms

- `include_default` 键用于指定是否包括默认的数据增强方法。例如，CIFAR100 数据集的默认数据增强方法定义在 [`fling/dataset/cifar100.py`](https://github.com/FLAIR-Community/Fling/blob/main/fling/dataset/cifar100.py) ，具体内容为：

```python
default_augmentation = dict(
    horizontal_flip=dict(p=0.5),
    random_rotation=dict(degree=15),
    Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    random_crop=dict(size=32, padding=4),
)
```

因此如果您正在使用 CIFAR100 数据集并设置 `include_default=True` ，那么您使用的实际数据变换将是：

```python
transforms=dict(
    horizontal_flip=dict(p=0.5),
    random_rotation=dict(degree=15),
    Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    random_crop=dict(size=32, padding=4),
)
```

请注意，对于不同的数据集，默认的数据增强可以是不同的，甚至可能为 None。

- 如果您想禁用默认的数据变换，只需使用 `include_default=False` 并定义您自己的方法：

```python
transforms=dict(
    include_default=False,
    # The following is optional.
    horizontal_flip=dict(p=0.5)
)
```

- 您还可以像这样覆写默认的变换：

```python
transforms=dict(
    include_default=True,
    # Overwrite
    horizontal_flip=dict(p=0.3)
)
```

对于 CIFAR100 来说，这段代码等同于：

```python
transforms=dict(
    horizontal_flip=dict(p=0.3),
    random_rotation=dict(degree=15),
    Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    random_crop=dict(size=32, padding=4),
)
```

- 关于详细的数据增强方法，请参考[此处](https://github.com/FLAIR-Community/Fling/blob/main/fling/utils/data_utils/data_transform.py)。