"""
    Dataset weighted random sampler.
"""

__all__ = ['WeightedRandomSampler']

import numpy as np
import mxnet as mx
from mxnet.gluon.data import Sampler


class WeightedRandomSampler(Sampler):
    """
    Samples elements from [0, length) randomly without replacement.

    Parameters
    ----------
    length : int
        Length of the sequence.
    weights : np.array of float
        Normalized weights of samples.
    """
    def __init__(self,
                 length,
                 weights):
        assert (isinstance(length, int) and length > 0)
        assert (len(weights) == length)
        assert (np.abs(weights.sum() - 1.0) <= 1e-5)
        self._length = length
        self._weights = weights.copy()

    def __iter__(self):
        indices = mx.nd.random.multinomial(mx.nd.array(self._weights), shape=self._length).asnumpy()
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self._length
