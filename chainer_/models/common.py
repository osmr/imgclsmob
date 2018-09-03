import chainer.functions as F
import chainer.links as L
from chainer import Chain

__all__ = ['SimpleSequential', 'conv1x1', 'SEBlock']


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


def conv1x1(in_channels,
            out_channels,
            stride=1,
            use_bias=True):
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

