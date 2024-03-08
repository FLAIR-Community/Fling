from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='cifar10', data_path='./data/CIFAR10', sample_method=dict(name='iid', train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0', local_eps=5, global_eps=50, batch_size=100, optimizer=dict(name='sgd', lr=0.1, momentum=0.9)
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='base_client', client_num=2),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=3, logging_path='./logging/cifar10_fedavg_visualization_resnet_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_visualization_pipeline
    generic_model_visualization_pipeline(exp_args, seed=0)
