from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='cifar10',
        data_path='./data/CIFAR10',
        sample_method=dict(name='dirichlet', alpha=1, train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=8,
        global_eps=40,
        batch_size=32,
        # learning rate for client
        lr=0.1,
        decay=1e-4,
        # learning rate for server
        server_lr=1,
        optimizer=dict(name='sgd', lr=0.1, momentum=0.9)
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    launcher=dict(name='serial', ),
    client=dict(name='scaffold_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(
        name='scaffold_group',
        aggregation_method='avg',
    ),
    other=dict(test_freq=3, logging_path='./logging/cifar10_scaffold_cnn_dirichlet_1')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline

    generic_model_pipeline(exp_args, seed=0)
