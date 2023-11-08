from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='cifar10',
        data_path='./data/cifar10',
        sample_method=dict(name='iid', train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=8,
        global_eps=40,
        batch_size=32,
        optimizer=dict(name='sgd', lr=0.02, momentum=0.9),
        # Only fine-tune parameters whose name contain the keyword "fc".
        finetune_parameters=dict(name='contain', keywords=['fc']),
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='base_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        aggregation_parameters=dict(
            name='all',
        ),
    ),
    other=dict(test_freq=1, logging_path='./logging/cifar10_fedavg_resnet8_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline

    generic_model_pipeline(exp_args, seed=0)
