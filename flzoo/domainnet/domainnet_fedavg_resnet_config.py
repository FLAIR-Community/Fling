from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='domainnet',
        data_path='./data/domainnet',
        sample_method=dict(name='cross_domain_iid', train_num=500, test_num=100),
        domains=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        base_dir='D:/VScode/FL/Data/DomainNet/'
    ),
    learn=dict(
        device='cuda:0', local_eps=8, global_eps=40, batch_size=32, optimizer=dict(name='sgd', lr=0.02, momentum=0.9)
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='base_client', client_num=40, num_users=5),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=3, logging_path='./logging/domainnet_fedavg_resnet_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import cross_domain_model_pipeline
    cross_domain_model_pipeline(exp_args, seed=0)
