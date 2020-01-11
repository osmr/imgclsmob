"""
    CondenseNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.
"""

__all__ = ['CondenseNet', 'condensenet74_c4_g4', 'condensenet74_c8_g8']

import os
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential, ChannelShuffle


class CondenseSimpleConv(Chain):
    """
    CondenseNet specific simple convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 groups):
        super(CondenseSimpleConv, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(size=in_channels)
            self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True,
                groups=groups)

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def condense_simple_conv3x3(in_channels,
                            out_channels,
                            groups):
    """
    3x3 version of the CondenseNet specific simple convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    """
    return CondenseSimpleConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=1,
        pad=1,
        groups=groups)


class CondenseComplexConv(Chain):
    """
    CondenseNet specific complex convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 groups):
        super(CondenseComplexConv, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(size=in_channels)
            self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True,
                groups=groups)
            self.c_shuffle = ChannelShuffle(
                channels=out_channels,
                groups=groups)

            self.index = initializers.generate_array(
                initializer=initializers._get_initializer(0),
                shape=(in_channels,),
                xp=self.xp,
                dtype=np.int32)
            self.register_persistent("index")

    def __call__(self, x):
        x = self.xp.take(x.array, self.index, axis=1)
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        x = self.c_shuffle(x)
        return x


def condense_complex_conv1x1(in_channels,
                             out_channels,
                             groups):
    """
    1x1 version of the CondenseNet specific complex convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    """
    return CondenseComplexConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=1,
        pad=0,
        groups=groups)


class CondenseUnit(Chain):
    """
    CondenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups):
        super(CondenseUnit, self).__init__()
        bottleneck_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bottleneck_size

        with self.init_scope():
            self.conv1 = condense_complex_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=groups)
            self.conv2 = condense_simple_conv3x3(
                in_channels=mid_channels,
                out_channels=inc_channels,
                groups=groups)

    def __call__(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.concat((identity, x), axis=1)
        return x


class TransitionBlock(Chain):
    """
    CondenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.
    """
    def __init__(self):
        super(TransitionBlock, self).__init__()
        with self.init_scope():
            self.pool = partial(
                F.average_pooling_2d,
                ksize=2,
                stride=2,
                pad=0)

    def __call__(self, x):
        x = self.pool(x)
        return x


class CondenseInitBlock(Chain):
    """
    CondenseNet specific initial block.

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
        super(CondenseInitBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=3,
                stride=2,
                pad=1,
                nobias=True)

    def __call__(self, x):
        x = self.conv(x)
        return x


class PostActivation(Chain):
    """
    CondenseNet final block, which performs the same function of postactivation as in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PostActivation, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(size=in_channels)
            self.activ = F.relu

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class CondenseLinear(Chain):
    """
    CondenseNet specific dense block.

    Parameters:
    ----------
    units : int
        Number of output channels.
    in_units : int
        Number of input channels.
    drop_rate : float
        Fraction of input channels for drop.
    """
    def __init__(self,
                 units,
                 in_units,
                 drop_rate=0.5):
        super(CondenseLinear, self).__init__()
        drop_in_units = int(in_units * drop_rate)
        with self.init_scope():
            self.dense = L.Linear(
                in_size=drop_in_units,
                out_size=units)

            self.index = initializers.generate_array(
                initializer=initializers._get_initializer(0),
                shape=(drop_in_units,),
                xp=self.xp,
                dtype=np.int32)
            self.register_persistent("index")

    def __call__(self, x):
        x = self.xp.take(x.array, self.index, axis=1)
        x = self.dense(x)
        return x


class CondenseNet(Chain):
    """
    CondenseNet model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    groups : int
        Number of groups in convolution layers.
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
                 groups,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(CondenseNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", CondenseInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        if i != 0:
                            setattr(stage, "trans{}".format(i + 1), TransitionBlock())
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), CondenseUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                groups=groups))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, 'post_activ', PostActivation(
                    in_channels=in_channels))
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", CondenseLinear(
                    units=classes,
                    in_units=in_channels))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_condensenet(num_layers,
                    groups=4,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".chainer", "models"),
                    **kwargs):
    """
    Create CondenseNet (converted) model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    groups : int
        Number of groups in convolution layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if num_layers == 74:
        init_block_channels = 16
        layers = [4, 6, 8, 10, 8]
        growth_rates = [8, 16, 32, 64, 128]
    else:
        raise ValueError("Unsupported CondenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(lambda xi, yi:
                      xi + [reduce(lambda xj, yj:
                                   xj + [xj[-1] + yj],
                                   [yi[1]] * yi[0],
                                   [xi[-1][-1]])[1:]],
                      zip(layers, growth_rates),
                      [[init_block_channels]])[1:]

    net = CondenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        groups=groups,
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


def condensenet74_c4_g4(**kwargs):
    """
    CondenseNet-74 (C=G=4) model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_condensenet(num_layers=74, groups=4, model_name="condensenet74_c4_g4", **kwargs)


def condensenet74_c8_g8(**kwargs):
    """
    CondenseNet-74 (C=G=8) model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_condensenet(num_layers=74, groups=8, model_name="condensenet74_c8_g8", **kwargs)


def _test():
    import chainer

    chainer.global_config.train = False

    pretrained = True

    models = [
        condensenet74_c4_g4,
        condensenet74_c8_g8,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != condensenet74_c4_g4 or weight_count == 4773944)
        assert (model != condensenet74_c8_g8 or weight_count == 2935416)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
