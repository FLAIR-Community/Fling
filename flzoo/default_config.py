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
