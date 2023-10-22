# 配置文件各字段含义

## 一个示例

在本文档中，我们提供了一个完整的配置文件，并解释了每个键的含义。

请注意，以下配置文件即为默认配置文件。换句话说，如果用户定义的配置中某些键不存在，将默认使用以下文件中的值。

以下是默认配置文件，您也可以在[这里](https://github.com/kxzxvbk/Fling/blob/main/argzoo/default_config.py)查看：

```python
import platform

default_exp_args = dict(
    # Configurations about data.
    data=dict(
        # Name for dataset. Must be registered in `DATASET_REGISTRY`.
        dataset='cifar10',
        # Root path for dataset.
        data_path='./data',
        # Image transformation methods, such as: Random Resized Crop(RRC), Resize, Color Jitter ...
        # The key ``include_default=True`` means that the default data-augmentation will be applied.
        transforms=dict(include_default=True),
        # How datasets distribute across all clients.
        sample_method=dict(
            name='iid',
            # Default training number for each client is 500.
            train_num=500,
            # Default testing number for each client is 500.
            test_num=100
        )
    ),
    # Configurations about learning process.
    learn=dict(
        # Running device for deep learning model. If only CPU is available, set this key to be "cpu".
        device='0',
        # Number of local epochs in each training round of each client.
        local_eps=8,
        # Number of global epochs (training rounds) in the total FL process.
        global_eps=40,
        # Batch size for local training, testing and fine-tuning.
        batch_size=32,
        # Test place for federated learning. Options: 'before_aggregation', 'after_aggregation'
        test_place=['after_aggregation'],
        # Optimizer used in local training.
        optimizer=dict(
            # Name for optimizer.
            name='sgd',
            # Learning rate of the optimizer.
            lr=0.02,
            # Momentum of the SGD optimizer.
            momentum=0.9
        ),
        # Learning rate scheduler. For each global epoch, use a dynamic learning rate.
        scheduler=dict(
            # Default to be "fix", which means learning rate used in each global epoch is identical.
            name='fix'
        ),
        # What parameters should be fine-tuned.
        finetune_parameters=dict(
            # For default case, every parameter should be fine-tuned.
            name='all'
        ),
    ),
    # Configurations about models.
    model=dict(
        # Name for model used. Must be registered in `MODEL_REGISTRY`.
        name='resnet8',
        # Arguments used in initializing corresponding model.
        # Channel of input image.
        input_channel=3,
        # Number of classes, i.e. the dimension for output logits.
        class_number=10,
    ),
    # Configurations about client.
    client=dict(
        # Name for client used. Must be registered in `CLIENT_REGISTRY`.
        name='base_client',
        # Number of clients.
        client_num=30,
        # The ratio of clients participated in each global epoch. For instance, if `sample_rate=0.5`,
        # only half of all clients will join federated learning in each global epoch.
        sample_rate=1,
        # The fraction ratio of test samples in total samples. For instance, if `val_frac=0.2`, this means
        # 20% of total data samples will be regarded as local validation dataset, and 80% for training dataset.
        val_frac=0,
    ),
    # Configurations about server.
    server=dict(
        # Name for server used. Must be registered in `SERVER_REGISTRY`.
        name='base_server'
    ),
    # Configurations about server.
    group=dict(
        # Name for group used. Must be registered in `GROUP_REGISTRY`.
        name='base_group',
        # How parameters in each client aggregate. Default to be "avg", which means a simple average.
        aggregation_method='avg',
        # What parameters in each client should be aggregated.
        aggregation_parameters=dict(
            # For default case, every parameter should be aggregated.
            name='all'
        ),
    ),
    # Launcher configurations.
    launcher=dict(
        # For the simplest launcher, serial is the suitable choice.
        name='serial'
        # If you want to use multiprocess to accelerate the training process, you can use the following setting.
        # name='multiprocessing',
        # num_proc=2
        # ``num_proc`` refers to the number of processes used in your program.
        # For the default setting, if your os is linux, the multiprocessing mode is enabled.
        # You can overwrite the default settings by yourself.
    ) if platform.system().lower() != 'linux' else dict(name='multiprocessing', num_proc=2),
    # Other configurations.
    other=dict(
        # Frequency for testing. For example, `test_freq=3` means the performance is tested every 3 global epochs.
        test_freq=3,
        # What is the logging directory of this experiment.
        # If the directory does not exist, it will be created automatically.
        # If the directory already exists, some parts might be over-written, which should be carefully inspected.
        logging_path='./logging/default_experiment',
        # The saved model checkpoint to start from. If it is set to ``None``, the training process
        # will start from scratch.
        resume_path=None,
        # Whether to print config is the command line.
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
    # This means to only fine-tune parameters whose names have keywords listed in `keywords`
    name='contain',
    keywords=['fc', 'bn']
)
```

相反，如果您想微调参数名称**不包含**关键字 "fc" 或 "bn" 的参数，可以这样写：

```python
finetune_parameters=dict(
    # This means to only fine-tune parameters except those whose names have keywords listed in `keywords`
    name='except',
    keywords=['fc', 'bn']
)
```

`group.aggregation_parameters` 控制在联邦学习框架下应该聚合哪些参数，对于它的设置和上面所述的 `learn.finetune_parameters` 是类似的。

### launcher

- 除了 "serial" 外，`launcher.name` 键还有其他用途。在除了 Linux 以外的操作系统上，默认配置 "serial" 意味着每个客户端上的所有操作都是串行执行的，这可能不够高效。我们可以使用多进程方法加速这个过程。
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

- `include_default` 键用于指定是否包括默认的数据增强方法。例如，如果您正在使用 CIFAR100 数据集并设置 `include_default=True` ，那么您使用的实际数据变换将是：

```python
transforms=dict(
    horizontal_flip=dict(p=0.5),
    random_rotation=dict(degree=15),
    Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    random_crop=dict(size=32, padding=4),
)
```

它在默认的数据增强中定义（[此处](https://github.com/kxzxvbk/Fling/blob/main/fling/dataset/cifar100.py)）。请注意，对于不同的数据集，默认的数据增强可以是不同的，甚至可能为 None。

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

- 关于详细的数据增强方法，请参考[此处](https://github.com/kxzxvbk/Fling/blob/main/fling/utils/data_utils/data_transform.py)。