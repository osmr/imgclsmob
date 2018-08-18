"""
    Common routines for models in Gluon.
"""

__all__ = ['ChannelShuffle', 'SEBlock']

from mxnet.gluon import nn, HybridBlock


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : NDArray
        Input tensor.
    groups : int
        Number of groups.
    """
    return x.reshape((0, -4, groups, -1, -2)).swapaxes(1, 2).reshape((0, -3, -2))


class ChannelShuffle(HybridBlock):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups,
                 **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.groups = groups

    def hybrid_forward(self, F, x):
        return channel_shuffle(x, self.groups)


class SEBlock(HybridBlock):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    """
    def __init__(self,
                 channels,
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        mid_cannels = channels // 16

        with self.name_scope():
            self.fc1 = nn.Dense(
                units=mid_cannels,
                use_bias=False,
                in_units=channels)
            self.relu = nn.Activation('relu')
            self.fc2 = nn.Dense(
                units=channels,
                use_bias=False,
                in_units=mid_cannels)
            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))
        return x

