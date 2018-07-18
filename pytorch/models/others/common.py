import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    """Channel Shuffle operation from ShuffleNet [arxiv: 1707.01083]

    Arguments:
        x (Tensor): tensor to shuffle.
        groups (int): groups to be split
    """
    batch, channels, height, width = x.size()
    if channels % groups != 0:
        raise ValueError('Tensor channels cannot be divided by `groups`.')
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

def split_by_size(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise ValueError("Sum of split sizes exceeds tensor dim")
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]
    return tuple(tensor.narrow(int(dim), int(start), int(length))
        for start, length in zip(splits, split_sizes))
