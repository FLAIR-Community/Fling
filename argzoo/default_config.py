default_exp_args = dict(
    data=dict(dataset='cifar10', data_path='./data', transforms=dict(), sample_method=dict(name='iid')),
    learn=dict(
        device='0',
        local_eps=8,
        global_eps=40,
        batch_size=32,
        loss='CrossEntropyLoss',
        optimizer=dict(name='sgd', lr=0.02, momentum=0.9),
        scheduler=dict(name='fix'),
        finetune_parameters=dict(name='all'),
    ),
    model=dict(
        name='cifar_resnet',
        input_channel=3,
        class_number=10,
    ),
    client=dict(
        name='base_client',
        client_num=30,
        sample_rate=1,
        test_frac=0,
    ),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        aggregation_parameters=dict(name='all'),
    ),
    other=dict(test_freq=3, logging_path='./logging/default_experiment')
)
