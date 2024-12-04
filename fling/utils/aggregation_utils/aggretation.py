from .fed_avg import fed_avg
from .fed_dis import fed_dis


def aggregate(method_name: str, *args, **kwargs):
    if method_name == 'avg':
        return fed_avg(*args, **kwargs)
    if method_name == 'fed_dis':
        return fed_dis(*args, **kwargs)
    raise ValueError(f"Unrecognized aggregation method: {method_name}")
