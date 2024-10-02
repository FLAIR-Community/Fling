from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='ag_news',
        data_path='./data/ag_news',
        sample_method=dict(name='iid', train_num=3000, test_num=300),
        max_length=512
    ),
    learn=dict(
        device='cuda:0', 
        local_eps=8, 
        global_eps=105,
        batch_size=256, 
        optimizer=dict(name='adam', lr=1e-3),
    ),
    model=dict(
        name='transformer_classifier',
        class_number=5,
        vocab_size=30333,
        n_layers=3
    ),
    client=dict(name='fedpart_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(
        name='fedpart_group',
        aggregation_method='avg'
    ),
    launcher=dict(name='serial'),
    other=dict(test_freq=1, logging_path='./logging/agnews_fedpart_transformer')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import partial_model_pipeline
    
    partial_model_pipeline(exp_args, seed=0)