import chainer.functions as F
import chainer.links as L
from chainer import Chain

__all__ = ['conv1x1', 'ChannelShuffle', 'SEBlock', 'SimpleSequential', 'DualPathSequential', 'Concurrent']


def conv1x1(in_channels,
            out_channels,
            stride=1,
            use_bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return L.Convolution2D(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        nobias=(not use_bias))


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : chainer.Variable or numpy.ndarray or cupy.ndarray
        Input variable.
    groups : int
        Number of groups.

    Returns
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted variable.
    """
    batch, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = F.reshape(x, shape=(batch, groups, channels_per_group, height, width))
    x = F.swapaxes(x, axis1=1, axis2=2)
    x = F.reshape(x, shape=(batch, channels, height, width))
    return x


class ChannelShuffle(Chain):
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
                 groups):
        super(ChannelShuffle, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def __call__(self, x):
        return channel_shuffle(x, self.groups)


class SEBlock(Chain):
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
                 reduction=16):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        with self.init_scope():
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_cannels,
                use_bias=True)
            self.conv2 = conv1x1(
                in_channels=mid_cannels,
                out_channels=channels,
                use_bias=True)

    def __call__(self, x):
        w = F.average_pooling_2d(x, ksize=x.shape[2:])
        w = self.conv1(w)
        w = F.relu(w)
        w = self.conv2(w)
        w = F.sigmoid(w)
        x = x * w
        return x


class SimpleSequential(Chain):
    """
    A sequential chain that can be used instead of Sequential.
    """
    def __init__(self):
        super(SimpleSequential, self).__init__()
        self.layer_names = []

    def __setattr__(self, name, value):
        super(SimpleSequential, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self.layer_names.append(name)

    def __delattr__(self, name):
        super(SimpleSequential, self).__delattr__(name)
        try:
            self.layer_names.remove(name)
        except ValueError:
            pass

    def __call__(self, x):
        for name in self.layer_names:
            x = self[name](x)
        return x


class DualPathSequential(SimpleSequential):
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
                 dual_path_scheme_ordinal=(lambda block, x1, x2: (block(x1), x2))):
        super(DualPathSequential, self).__init__()
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def __call__(self, x1, x2=None):
        length = len(self.layer_names)
        for i, block_name in enumerate(self.layer_names):
            block = self[block_name]
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(block, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(block, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class Concurrent(SimpleSequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=1):
        super(Concurrent, self).__init__()
        self.axis = axis

    def __call__(self, x):
        out = []
        for name in self.layer_names:
            out.append(self[name](x))
        out = F.concat(tuple(out), axis=self.axis)
        return out
