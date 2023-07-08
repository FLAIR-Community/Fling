from easydict import EasyDict

exp_args = dict(
    data=dict(dataset='cifar10', data_path='./data/CIFAR10', sample_method=dict(name='dirichlet', alpha=0.5, train_num=500, test_num=100)),
    learn=dict(
        device='cuda:0',
        local_eps=5,
        local_lora_eps=2,
        global_eps=200,
        batch_size=100,
        optimizer=dict(name='sgd', lr=0.1, momentum=0.9),
        # Only fine-tune parameters whose name contain the keyword "fc".
        finetune_parameters=dict(name='contain', keywords=['lora_A', 'lora_B']),
    ),
    model=dict(
        name='lora_resnet8',
        input_channel=3,
        class_number=10,
        Conv_r=1,
        Linear_r=3,
        lora_alpha=1,
    ),
    client=dict(name='fedlora_client', client_num=40, val_frac=0),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        # Only aggregate parameters whose name does not contain the keyword "fc".
        aggregation_parameters=dict(
            name='except',
            keywords=['lora_A', 'lora_B'],
        ),
    ),
    other=dict(test_freq=1, logging_path='./logging/cifar10_fedlora_resnet_dirichlet_05')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_serial_pipeline

    personalized_model_serial_pipeline(exp_args, seed=0)
