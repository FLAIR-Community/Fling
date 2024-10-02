from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='cifar100', data_path='./data/CIFAR100', sample_method=dict(name='iid', train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0', local_eps=8, global_eps=141, batch_size=32,
        optimizer=dict(name='adam', lr=1e-3),
        test_place=['before_aggregation', 'after_aggregation'],
        finetune_parameters=dict(name='contain', keywords=['fc']),
    ),
    model=dict(
        name='resnet18',
        input_channel=3,
        class_number=100,
    ),
    client=dict(name='fedpart_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(name='fedpart_group', aggregation_method='avg'),
    other=dict(test_freq=1, logging_path='./logging/cifar100_fedpart_resnet18_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import partial_model_pipeline

    partial_model_pipeline(exp_args, seed=0)
