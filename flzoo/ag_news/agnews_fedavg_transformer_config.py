from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='ag_news',
        data_path='./data/ag_news',
        sample_method=dict(name='dirichlet', alpha=1, train_num=0, test_num=0),
        max_length=512
    ),
    learn=dict(
        device='cuda:0', local_eps=8, global_eps=40, batch_size=256, optimizer=dict(name='sgd', lr=0.02, momentum=0.9)
    ),
    model=dict(
        name='transformer_classifier',
        class_number=5,
        vocab_size=30333,
    ),
    client=dict(name='base_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    launcher=dict(name='serial'),
    other=dict(test_freq=1, logging_path='./logging/agnews_fedavg_transformer')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=0)
