from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='sogou_news',
        data_path='./data/sogou_news',
        sample_method=dict(name='dirichlet', alpha=1, train_num=0, test_num=0),
        max_length=512
    ),
    learn=dict(
        device='cuda:0',
        local_eps=8,
        global_eps=40,
        batch_size=256,
        optimizer=dict(name='sgd', lr=0.02, momentum=0.9),
        finetune_parameters=dict(name='contain', keywords=['classifier']),
    ),
    model=dict(
        name='transformer_classifier',
        class_number=5,
        vocab_size=10,
    ),
    client=dict(name='base_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        aggregation_parameters=dict(
            name='except',
            keywords=['classifier'],
        ),
    ),
    other=dict(test_freq=3, logging_path='./logging/sogou_news_fedper_transformer')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline

    personalized_model_pipeline(exp_args, seed=0)
