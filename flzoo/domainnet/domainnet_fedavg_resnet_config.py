from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='domainnet',
        data_path='./data/',
        domains='clipart,infograph,painting,quickdraw,real,sketch',
        transforms=dict(
            include_default=False,
            resize=dict(size=(256, 256)),
            horizontal_flip=dict(p=0.5),
            random_rotation=dict(degree=(-30, 30)),
        ),
        sample_method=dict(name='iid', train_num=500, test_num=100000)
    ),
    learn=dict(
        device='cuda:0', local_eps=1, global_eps=200, batch_size=64, optimizer=dict(name='sgd', lr=0.01, momentum=0.5)
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='cross_domain_client', client_num=1),
    server=dict(name='cross_domain_server'),
    group=dict(name='cross_domain_group', aggregation_method='avg'),
    other=dict(test_freq=3, logging_path='./logging/domainnet_fedavg_resnet_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import cross_domain_pipeline

    cross_domain_pipeline(exp_args, seed=0)
