from easydict import EasyDict

lamda = 1
alphaK = 10000
sigma = 100
decay_rate = 0.1
decay_frequency = 30
noniid = 'dirichlet'
alpha = 0.1
seed = 2


exp_args = dict(
    data=dict(
        dataset='cifar10',
        data_path='./data/CIFAR10',
        sample_method=dict(name=noniid, alpha=alpha, train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=5,
        global_eps=300,
        batch_size=100,
        optimizer=dict(name='sgd', lr=0.1, momentum=0.9),
        finetune_parameters=dict(name='all'),
        test_place=['after_aggregation'],
        lamda=lamda,  # regularization weight for FedAMP
        alphaK=alphaK,  # lambda/sqrt(GLOABL-ITRATION) according to the paper
        sigma=sigma,    # hyperparameter in function A
        decay_rate=decay_rate,  # decay rate of alphaK in FedAMP
        decay_frequency=decay_frequency,  # decay frequency of alphaK in FedAMP
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='fedamp_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(
        name='fedamp_group',
    ),
    other=dict(test_freq=1, logging_path=f'./logging/cifar10_fedamp_resnet/{noniid}_{alpha}/{lamda}_{alphaK}_{sigma}/{seed}')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline

    personalized_model_pipeline(exp_args, seed=seed)
