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
        global_eps=125,
        batch_size=32,
        optimizer=dict(name='adam', lr=1e-3),
        test_place=['before_aggregation', 'after_aggregation'],
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
        include_non_param=False,
    ),
    other=dict(test_freq=1, logging_path='./logging/cifar10_fedpart_resnet8_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    # Here, `partial_model_pipeline` can be replaced with `generic_model_pipeline`
    # to run the full-parameter training (FedAvg) with the corresponding configuration.
    # Please note that the `other.logging_path` above should be modified accordingly.
    from fling.pipeline import partial_model_pipeline

    partial_model_pipeline(exp_args, seed=0)
