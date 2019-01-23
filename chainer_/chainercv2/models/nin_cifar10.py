"""
    NIN for CIFAR-10, implemented in Chainer.
    Original paper: 'Network In Network,' https://arxiv.org/abs/1312.4400.
"""

__all__ = ['CIFAR10NIN', 'nin_cifar10']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


class NINConv(Chain):
    """
    NIN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 pad=0):
        super(NINConv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=False)
            self.activ = F.relu

    def __call__(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class CIFAR10NIN(Chain):
    """
    NIN model for CIFAR-10 from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_ksizes : list of int
        Convolution window sizes for the first units in each stage.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 first_ksizes,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFAR10NIN, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            if (j == 0) and (i != 0):
                                if i == 1:
                                    setattr(stage, "pool{}".format(i + 1), partial(
                                        F.max_pooling_2d,
                                        ksize=3,
                                        stride=2,
                                        pad=1,
                                        cover_all=False))
                                else:
                                    setattr(stage, "pool{}".format(i + 1), partial(
                                        F.average_pooling_2d,
                                        ksize=3,
                                        stride=2,
                                        pad=1))
                                setattr(stage, "dropout{}".format(i + 1), partial(
                                    F.dropout,
                                    ratio=0.5))
                            kernel_size = first_ksizes[i] if j == 0 else 1
                            padding = (kernel_size - 1) // 2
                            setattr(stage, "unit{}".format(j + 1), NINConv(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                ksize=kernel_size,
                                pad=padding))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, 'final_conv', NINConv(
                    in_channels=in_channels,
                    out_channels=classes,
                    ksize=1))
                setattr(self.output, 'final_pool', partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))
                setattr(self.output, "final_flatten", partial(
                    F.reshape,
                    shape=(-1, classes)))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_nin_cifar10(model_name=None,
                    pretrained=False,
                    root=os.path.join('~', '.chainer', 'models'),
                    **kwargs):
    """
    Create NIN model for CIFAR-10 with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    channels = [[192, 160, 96], [192, 192, 192], [192, 192]]
    first_ksizes = [5, 5, 3]

    net = CIFAR10NIN(
        channels=channels,
        first_ksizes=first_ksizes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def nin_cifar10(**kwargs):
    """
    NIN model for CIFAR-10 from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_nin_cifar10(model_name="nin_cifar10", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        nin_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nin_cifar10 or weight_count == 966986)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, 10))


if __name__ == "__main__":
    _test()
