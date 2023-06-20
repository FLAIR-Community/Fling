from utils.registry_utils import CLIENT_REGISTRY


def get_client(train_dataset, args, client_id):
    return CLIENT_REGISTRY.build(args.client.name, train_dataset, args, client_id)
