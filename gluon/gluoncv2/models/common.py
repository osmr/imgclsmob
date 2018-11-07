"""
    Common routines for models in Gluon.
"""

__all__ = ['conv1x1', 'ChannelShuffle', 'ChannelShuffle2', 'SEBlock', 'DualPathSequential', 'ParametricSequential',
           'ParametricConcurrent']

from mxnet.gluon import nn, HybridBlock


def conv1x1(in_channels,
            out_channels,
            strides=1,
            use_bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        strides=strides,
        use_bias=use_bias,
        in_channels=in_channels)


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

    Returns
    -------
    NDArray
        Resulted tensor.
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


def channel_shuffle2(x,
                     channels_per_group):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.

    Parameters:
    ----------
    x : NDArray
        Input tensor.
    channels_per_group : int
        Number of channels per group.

    Returns
    -------
    NDArray
        Resulted tensor.
    """
    return x.reshape((0, -4, channels_per_group, -1, -2)).swapaxes(1, 2).reshape((0, -3, -2))


class ChannelShuffle2(HybridBlock):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.

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
        super(ChannelShuffle2, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.channels_per_group = channels // groups

    def hybrid_forward(self, F, x):
        return channel_shuffle2(x, self.channels_per_group)


class SEBlock(HybridBlock):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        mid_cannels = channels // reduction

        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_cannels,
                use_bias=True)
            self.relu = nn.Activation('relu')
            self.conv2 = conv1x1(
                in_channels=mid_cannels,
                out_channels=channels,
                use_bias=True)
            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = F.broadcast_mul(x, w)
        return x


class DualPathSequential(nn.HybridSequential):
    """
    A sequential container for hybrid blocks with dual inputs/outputs.
    Blocks will be executed in the order they are added.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first blocks with single input/output.
    last_ordinals : int, default 0
        Number of the final blocks with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a block.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal block.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda block, x1, x2: block(x1, x2)),
                 dual_path_scheme_ordinal=(lambda block, x1, x2: (block(x1), x2)),
                 **kwargs):
        super(DualPathSequential, self).__init__(**kwargs)
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def hybrid_forward(self, F, x1, x2=None):
        length = len(self._children.values())
        for i, block in enumerate(self._children.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(block, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(block, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class ParametricSequential(nn.HybridSequential):
    """
    A sequential container for modules with parameters.
    Blocks will be executed in the order they are added.
    """
    def __init__(self,
                 **kwargs):
        super(ParametricSequential, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        for block in self._children.values():
            x = block(x, *args, **kwargs)
        return x


class ParametricConcurrent(nn.HybridSequential):
    """
    A container for concatenation of modules with parameters.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self,
                 axis=1,
                 **kwargs):
        super(ParametricConcurrent, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = []
        for block in self._children.values():
            out.append(block(x, *args, **kwargs))
        out = F.concat(*out, dim=self.axis)
        return out
