from utils.registry_utils import GROUP_REGISTRY


def get_group(args, logger):
    return GROUP_REGISTRY.build(args.group.name, args, logger)
