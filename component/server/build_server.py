from utils.registry_utils import SERVER_REGISTRY


def get_server(args, test_dataset):
    return SERVER_REGISTRY.build(args.server.name, args, test_dataset)
