from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='domainnet',
        data_path='./data/domainnet',
        sample_method=dict(name='cross_domain_iid', train_num=500, test_num=30000),
        domains='clipart,infograph,painting,quickdraw,real,sketch',
        base_dir='D:/My_Codes/Federate-Learning/Data/DomainNet/'  
    ),
    learn=dict(
        device='cuda:0', 
        local_eps=1, 
        global_eps=500, 
        batch_size=64, 
        optimizer=dict(name='sgd', lr=0.01, momentum=0.5)
    ),
    model=dict(
        name='resnet18',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='cross_domain_base_client', client_num=1),
    server=dict(name='cross_domain_base_server'),
    group=dict(name='cross_domain_base_group', aggregation_method='avg'),
    # add group personal choice: Only aggregate parameters whose name does not contain the keyword "fc".
        # aggregation_parameters=dict(name='except', keywords=['fc']
    other=dict(test_freq=5, logging_path='./logging/cross_domain_domainnet_fedavg_resnet_iid_500')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline.cross_domain_model_pipeline import cross_domain_model_pipeline
    cross_domain_model_pipeline(exp_args, seed=0)
