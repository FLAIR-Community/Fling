import copy

from utils.registry_utils import MODEL_REGISTRY


def get_model(args):
    args = copy.deepcopy(args)
    model_name = args.model.pop('name')
    return MODEL_REGISTRY(model_name, **args.model)
