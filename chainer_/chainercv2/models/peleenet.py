"""
    PeleeNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Pelee: A Real-Time Object Detection System on Mobile Devices,' https://arxiv.org/abs/1804.06882.
"""

__all__ = ['PeleeNet', 'peleenet']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, Concurrent, SimpleSequential


class PeleeBranch1(Chain):
    """
    PeleeNet branch type 1 block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the second convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 stride=1):
        super(PeleeBranch1, self).__init__()
        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                stride=stride)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PeleeBranch2(Chain):
    """
    PeleeNet branch type 2 block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(PeleeBranch2, self).__init__()
        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels)
            self.conv3 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class StemBlock(Chain):
    """
    PeleeNet stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(StemBlock, self).__init__()
        mid1_channels = out_channels // 2
        mid2_channels = out_channels * 2

        with self.init_scope():
            self.first_conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2)

            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", PeleeBranch1(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    mid_channels=mid1_channels,
                    stride=2))
                setattr(self.branches, "branch2", partial(
                    F.max_pooling_2d,
                    ksize=2,
                    stride=2,
                    pad=0,
                    cover_all=False))

            self.last_conv = conv1x1_block(
                in_channels=mid2_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.first_conv(x)
        x = self.branches(x)
        x = self.last_conv(x)
        return x


class DenseBlock(Chain):
    """
    PeleeNet dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bottleneck_size : int
        Bottleneck width.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bottleneck_size):
        super(DenseBlock, self).__init__()
        inc_channels = (out_channels - in_channels) // 2
        mid_channels = inc_channels * bottleneck_size

        with self.init_scope():
            self.branch1 = PeleeBranch1(
                in_channels=in_channels,
                out_channels=inc_channels,
                mid_channels=mid_channels)
            self.branch2 = PeleeBranch2(
                in_channels=in_channels,
                out_channels=inc_channels,
                mid_channels=mid_channels)

    def __call__(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = F.concat((x, x1, x2), axis=1)
        return x


class TransitionBlock(Chain):
    """
    PeleeNet's transition block, like in DensNet, but with ordinary convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(TransitionBlock, self).__init__()
        with self.init_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels)
            self.pool = partial(
                F.average_pooling_2d,
                ksize=2,
                stride=2,
                pad=0)

    def __call__(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PeleeNet(Chain):
    """
    PeleeNet model from 'Pelee: A Real-Time Object Detection System on Mobile Devices,'
    https://arxiv.org/abs/1804.06882.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck_sizes : list of int
        Bottleneck sizes for each stage.
    dropout_rate : float, default 0.5
        Parameter of Dropout layer. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck_sizes,
                 dropout_rate=0.5,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(PeleeNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", StemBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    bottleneck_size = bottleneck_sizes[i]
                    stage = SimpleSequential()
                    with stage.init_scope():
                        if i != 0:
                            setattr(stage, "trans{}".format(i + 1), TransitionBlock(
                                in_channels=in_channels,
                                out_channels=in_channels))
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), DenseBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                bottleneck_size=bottleneck_size))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_block", conv1x1_block(
                    in_channels=in_channels,
                    out_channels=in_channels))
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "dropout", partial(
                    F.dropout,
                    ratio=dropout_rate))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_peleenet(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".chainer", "models"),
                 **kwargs):
    """
    Create PeleeNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    growth_rate = 32
    layers = [3, 4, 8, 6]
    bottleneck_sizes = [1, 2, 4, 4]

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1]])[1:]],
        layers,
        [[init_block_channels]])[1:]

    net = PeleeNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck_sizes=bottleneck_sizes,
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


def peleenet(**kwargs):
    """
    PeleeNet model from 'Pelee: A Real-Time Object Detection System on Mobile Devices,'
    https://arxiv.org/abs/1804.06882.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_peleenet(model_name="peleenet", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        peleenet,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != peleenet or weight_count == 2802248)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
