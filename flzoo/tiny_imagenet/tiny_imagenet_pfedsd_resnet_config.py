from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='tiny_imagenet',
        data_path='./data/tiny-imagenet-200',
        sample_method=dict(name='dirichlet', alpha=0.5, train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=5,
        global_eps=200,
        batch_size=100,
        optimizer=dict(name='sgd', lr=0.01, momentum=0.9),
        finetune_parameters=dict(name='all'),
        test_place=['before_aggregation'],
        lamda=0.5,  # the weight of self distillation loss
        tau=3,  # the temperature of softmax in self distillation loss
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=200,
    ),
    client=dict(name='pfedsd_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        # Only aggregate parameters whose name does not contain the keyword "fc".
        aggregation_parameters=dict(name='all', ),
    ),
    other=dict(test_freq=1, logging_path='./logging/tiny_imagenet_pfedsd_resnet_dirichlet_05')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline

    personalized_model_pipeline(exp_args, seed=0)
