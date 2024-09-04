from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='mnist',
        data_path='./data/mnist',
        sample_method=dict(name='dirichlet', alpha=1.0, train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=8,
        global_eps=40,
        batch_size=32,
        lr=0.08,
        decay=1e-4,
        server_lr=1,
        optimizer=dict(name='sgd', lr=0.1, momentum=0.9)
    ),
    model=dict(
        name='cnn',
        input_channel=1,
        class_number=10,
    ),
    launcher=dict(name='serial'),
    client=dict(name='scaffold_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(name='scaffold_group', aggregation_method='avg'),
    other=dict(test_freq=3, logging_path='./logging/emnist_scaffold_cnn_iid_demo')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline

    generic_model_pipeline(exp_args, seed=0)
