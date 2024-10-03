from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='tiny_imagenet',
        data_path='./data/tiny-imagenet-200',
        sample_method=dict(name='iid', train_num=500, test_num=100)
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
        class_number=200,
    ),
    client=dict(name='base_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg', include_non_param=False,),
    other=dict(test_freq=1, logging_path='./logging/tiny_imagenet_fedpart_resnet18_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import partial_model_pipeline

    partial_model_pipeline(exp_args, seed=0)
