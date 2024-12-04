from typing import List, Callable

from torch import Tensor


def tensor_reduce(func: Callable, sequence: List[Tensor], device: str) -> Tensor:
    """
    Overview:
        Reduce a sequence of tensors on a given device.
    Arguments:
        - func: The callable function to reduce this list of tensors. For example, if the given ``func`` is
            ``lamda x, y: x + y``. This function will return the sum of all tensors in the given ``sequence``.
        - sequence: The given list of tensors.
        - device: The device to operate this reduction on.
    Returns:
        - ret: The reduced tensor using given function and sequence.
    """
    assert len(sequence) > 0, f"Invalid length when calling reduce. Got: {len(sequence)}"
    orig_device = sequence[0].device
    ret = sequence[0].clone().to(device)
    for i in range(1, len(sequence)):
        ret = func(ret, sequence[i].to(device))
    ret = ret.to(orig_device)
    return ret
