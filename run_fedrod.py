from copy import deepcopy
from fling.pipeline import personalized_model_pipeline


def run_3seed(exp_args):
    personalized_model_pipeline(exp_args, seed=0)
    personalized_model_pipeline(exp_args, seed=1)
    personalized_model_pipeline(exp_args, seed=2)


def launch_diri(algo_name, dataset_name, base_args, alphas):
    base_args = deepcopy(base_args)
    base_args.data.sample_method.name = 'dirichlet'

    for alpha in alphas:
        base_args.data.sample_method.alpha = alpha
        base_args.other.logging_path = f'./logging/{dataset_name}_{algo_name}_dirichlet_{alpha}'
        run_3seed(base_args)


def launch_patho(algo_name, dataset_name, base_args, alphas):
    base_args = deepcopy(base_args)
    base_args.data.sample_method.name = 'pathological'

    for alpha in alphas:
        base_args.data.sample_method.alpha = alpha
        base_args.other.logging_path = f'./logging/{dataset_name}_{algo_name}_pathological_{alpha}'
        run_3seed(base_args)


def main():
    # cifar10
    from argzoo.cifar10.cifar10_fedrod_resnet_config import exp_args
    launch_diri(algo_name='fedrod', dataset_name='cifar10', base_args=exp_args, alphas=[0.1, 0.5, 1.0])
    launch_patho(algo_name='fedrod', dataset_name='cifar10', base_args=exp_args, alphas=[2])

    # cifar100
    from argzoo.cifar100.cifar100_fedrod_resnet_config import exp_args
    launch_diri(algo_name='fedrod', dataset_name='cifar100', base_args=exp_args, alphas=[0.1, 0.5, 1.0])
    launch_patho(algo_name='fedrod', dataset_name='cifar100', base_args=exp_args, alphas=[2])

    # tiny
    from argzoo.imagenet_tiny.tiny_imagenet_fedrod_resnet_config import exp_args
    launch_diri(algo_name='fedrod', dataset_name='tiny', base_args=exp_args, alphas=[0.1, 0.5, 1.0])
    launch_patho(algo_name='fedrod', dataset_name='tiny', base_args=exp_args, alphas=[2])

