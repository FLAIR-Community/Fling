# Meanings for Each Configuration Key

## An Example

In this document, we provide a simple configuration file and explain the meanings of each key.

```
default_exp_args = dict(
    # Configurations about data.
    data=dict(
        # Name for dataset. Must be registered in `DATASET_REGISTRY`.
        dataset='cifar10',
        # Root path for dataset.
        data_path='./data',
        # Image transformation methods, such as: Random Resized Crop(RRC), Resize, Color Jitter ...
        transforms=dict(),
        # How datasets distribute across all clients.
        sample_method=dict(name='iid')
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
        # TODO
        loss='CrossEntropyLoss',
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
        name='cifar_resnet',
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
        # The fraction ratio of test samples in total samples. For instance, if `test_frac=0.2`, this means
        # 20% of total data samples will be regarded as local test dataset, and 80% for training dataset.
        test_frac=0,
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
    # Other configurations.
    other=dict(
        # Frequency for testing. For example, `test_freq=3` means the performance is tested every 3 global epochs.
        test_freq=3,
        # What is the logging directory of this experiment.
        # If the directory does not exist, it will be created automatically.
        # If the directory already exists, some parts might be over-written, which should be carefully inspected.
        logging_path='./logging/default_experiment'
    )
)
```

## Other usages

- Beside "fix", the key `learn.scheduler` has a variety of types. For example, we can define a scheduler using cosine function:

```
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

- Beside "all", the key `learn.finetune_parameters` has other usages. For example, if you only want to fine-tune parameters whose names **contain** keywords "fc" or "bn", you can write:

```
finetune_parameters=dict(
    # This means to only fine-tune parameters whose names have keywords listed in `keywords`
    name='contain',
    keywords=['fc', 'bn']
)
```

On the opposite, if you want to fine-tune parameters whose names **except** keywords "fc" or "bn", you can write:

```
finetune_parameters=dict(
    # This means to only fine-tune parameters except those whose names have keywords listed in `keywords`
    name='except',
    keywords=['fc', 'bn']
)
```

The setting for `group.aggregation_parameters` is similar.