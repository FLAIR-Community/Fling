# Meanings for Each Configuration Key

## Default Configuration Instance

In this document, we provide a complete configuration file and explain the meanings of each key.

Note that the following configuration file is exactly the default configuration file. In other words, if some keys do not exist in user-defined config, the value in the following file will be used by default.

Here is the default configuration file, you can also view it at [here](https://github.com/FLAIR-Community/Fling/blob/main/argzoo/default_config.py) :

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
        # Whether non-parameter weights will be globally aggregated. Default to be ``True``, which means
        # all weights in ``model.state_dict()`` will be globally aggregated.
        include_non_param=True,
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

## Other tutorials

### learn.scheduler

This key controls the learning rate scheduler. Beside "fix", the key `learn.scheduler` has a variety of types. For example, we can define a scheduler using cosine function:

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

For other types schedulers, please refer to `fling.utils.LRScheduler`.

### learn.finetune_parameters

This key controls what parameters should be finetuned. Beside "all", the key `learn.finetune_parameters` has other values. For example, if you only want to fine-tune parameters whose names **contain** keywords "fc" or "bn", you can write:

```python
finetune_parameters=dict(
    # This means to only fine-tune parameters whose names have keywords listed in `keywords`.
    name='contain',
    keywords=['fc', 'bn']
)
```

On the opposite, if you want to fine-tune parameters whose names **except** keywords "fc" or "bn", you can write:

```python
finetune_parameters=dict(
    # This means to only fine-tune parameters except those whose names have keywords listed in `keywords`.
    name='except',
    keywords=['fc', 'bn']
)
```

The setting for `group.aggregation_parameters` , which controls what parameters should be aggregated under the framework of Federated Learning, is similar.

To show the differences in meaning and usage between the parameters `learn.finetune_parameters` and `group.aggregation_parameters`, let's take the configuration file of the **FedPer** algorithm, [`cifar10_fedper_resnet_config.py`](https://github.com/FLAIR-Community/Fling/blob/main/fling/dataset/cifar100.py), as an example. The definitions of these two parameters in the configuration file are as follows:

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

In this setup for Federated Learning, the `group.aggregation_parameters` allows all clients to share parameters except those containing the "fc" keyword, while the `learn.finetune_parameters` parameter enables each client to fine-tune its own parameters that contain the "fc" keyword.

### launcher

- Beside "serial", the key `launcher.name` has other modes. On os other than Linux, the default configuration "serial" means that all operations on each client is executed serially, which can be not efficient enough. We can use the multiprocessing method to accelerate this process.
- On Linux system, the default `launcher.name` is "multiprocessing" and number of processes `num_proc=2`:

```python
launcher=dict(
    name='multiprocessing',
    num_proc=2
)
```

There are several things important:

1) For Windows users, this multiprocessing can be not stable due to the fact that multiprocessing + PyTorch does not support for Windows well. It may lead to unexpected bugs. If you meet any bugs related to multiprocessing on Windows, consider using the serial launcher instead.
2) Due to the new features of PyTorch 2.0, the compiled model does not currently support multiprocessing execution. In our design, if you enable the multiprocessing mode, the compilation of PyTorch will be disabled. This bug might be solved by PyTorch after this pr lands: https://github.com/pytorch/pytorch/pull/101651
3) The `num_proc` refers to the number of processes used in the program. The optimal value of this value can be affected by the task you execute, the hardware you use, the neural network you employ, the data you use ...... You should tune this value carefully on your system and choose the best choice. Here is a simple benchmark on A100 with 4-core CPU, batch_size=32:

| num_proc          | 1     | 2     | 3     | 4     | 8     |
| ----------------- | ----- | ----- | ----- | ----- | ----- |
| MNIST + CNN       | 21.7s | 14.9s | 15.3s | 16.5s | 25.2s |
| CIFAR10 + ResNet8 | 45.2s | 32.1s | 30.1s | 32.5s | 40.4s |

### data.transforms

- The key `include_default` refers to whether include the default data augmentation methods. For example, if you are using the CIFAR100 dataset and set `include_default=True`. The exact data transforms you use is

```python
transforms=dict(
    horizontal_flip=dict(p=0.5),
    random_rotation=dict(degree=15),
    Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    random_crop=dict(size=32, padding=4),
)
```

, which is defined in the default augmentation [`fling/dataset/cifar100.py`](https://github.com/FLAIR-Community/Fling/blob/main/fling/dataset/cifar100.py). Note that for different datasets, the default augmentations can be different and can be even None.

- If users want to disable the default transforms, just use `include_default=False` and define your own methods:

```python
transforms=dict(
    include_default=False,
    # The following is optional.
    horizontal_flip=dict(p=0.5)
)
```

- Users can also overwrite the default transforms like this:

```python
transforms=dict(
    include_default=True,
    # Overwrite
    horizontal_flip=dict(p=0.3)
)
```

For CIFAR100, this code is equivalent to:

```python
transforms=dict(
    horizontal_flip=dict(p=0.3),
    random_rotation=dict(degree=15),
    Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    random_crop=dict(size=32, padding=4),
)
```

- For detailed data augmentation methods, please refer to [here](https://github.com/FLAIR-Community/Fling/blob/main/fling/utils/data_utils/data_transform.py) .

