from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='mnist',
        data_path='./data/mnist',
        sample_method=dict(name='dirichlet', alpha=0.2, train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=1,
        global_eps=4,
        batch_size=32,
        optimizer=dict(name='sgd', lr=0.02, momentum=0.9),
        # Only fine-tune parameters whose name contain the keyword "fc".
        finetune_parameters=dict(name='contain', keywords=['fc']),
    ),
    model=dict(
        name='cnn',
        input_channel=1,
        class_number=10,
    ),
    client=dict(name='base_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        # Only aggregate parameters whose name does not contain the keyword "fc".
        aggregation_parameters=dict(
            name='except',
            keywords=['fc'],
        ),
    ),
    other=dict(test_freq=3, logging_path='./logging/mnist_fedper_cnn_dirichlet_02_demo')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline

    personalized_model_pipeline(exp_args, seed=0)
