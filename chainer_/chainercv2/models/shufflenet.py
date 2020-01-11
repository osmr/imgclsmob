"""
    ShuffleNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.
"""

__all__ = ['ShuffleNet', 'shufflenet_g1_w1', 'shufflenet_g2_w1', 'shufflenet_g3_w1', 'shufflenet_g4_w1',
           'shufflenet_g8_w1', 'shufflenet_g1_w3d4', 'shufflenet_g3_w3d4', 'shufflenet_g1_wd2', 'shufflenet_g3_wd2',
           'shufflenet_g1_wd4', 'shufflenet_g3_wd4']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv3x3, depthwise_conv3x3, SimpleSequential, ChannelShuffle


class ShuffleUnit(Chain):
    """
    ShuffleNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups in convolution layers.
    downsample : bool
        Whether do downsample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 downsample,
                 ignore_group):
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        with self.init_scope():
            self.compress_conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=(1 if ignore_group else groups))
            self.compress_bn1 = L.BatchNormalization(size=mid_channels)
            self.c_shuffle = ChannelShuffle(
                channels=mid_channels,
                groups=groups)
            self.dw_conv2 = depthwise_conv3x3(
                channels=mid_channels,
                stride=(2 if self.downsample else 1))
            self.dw_bn2 = L.BatchNormalization(size=mid_channels)
            self.expand_conv3 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                groups=groups)
            self.expand_bn3 = L.BatchNormalization(size=out_channels)
            if downsample:
                self.avgpool = partial(
                    F.average_pooling_2d,
                    ksize=3,
                    stride=2,
                    pad=1)
            self.activ = F.relu

    def __call__(self, x):
        identity = x
        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        x = self.dw_conv2(x)
        x = self.dw_bn2(x)
        x = self.expand_conv3(x)
        x = self.expand_bn3(x)
        if self.downsample:
            identity = self.avgpool(identity)
            x = F.concat((x, identity), axis=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


class ShuffleInitBlock(Chain):
    """
    ShuffleNet specific initial block.

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
        super(ShuffleInitBlock, self).__init__()
        with self.init_scope():
            self.conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2)
            self.bn = L.BatchNormalization(size=out_channels)
            self.activ = F.relu
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class ShuffleNet(Chain):
    """
    ShuffleNet model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

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
        super(ShuffleNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", ShuffleInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            downsample = (j == 0)
                            ignore_group = (i == 0) and (j == 0)
                            setattr(stage, "unit{}".format(j + 1), ShuffleUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                groups=groups,
                                downsample=downsample,
                                ignore_group=ignore_group))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_shufflenet(groups,
                   width_scale,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".chainer", "models"),
                   **kwargs):
    """
    Create ShuffleNet model with specific parameters.

    Parameters:
    ----------
    groups : int
        Number of groups in convolution layers.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 24
    layers = [4, 8, 4]

    if groups == 1:
        channels_per_layers = [144, 288, 576]
    elif groups == 2:
        channels_per_layers = [200, 400, 800]
    elif groups == 3:
        channels_per_layers = [240, 480, 960]
    elif groups == 4:
        channels_per_layers = [272, 544, 1088]
    elif groups == 8:
        channels_per_layers = [384, 768, 1536]
    else:
        raise ValueError("The {} of groups is not supported".format(groups))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    net = ShuffleNet(
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


def shufflenet_g1_w1(**kwargs):
    """
    ShuffleNet 1x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=1, width_scale=1.0, model_name="shufflenet_g1_w1", **kwargs)


def shufflenet_g2_w1(**kwargs):
    """
    ShuffleNet 1x (g=2) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=2, width_scale=1.0, model_name="shufflenet_g2_w1", **kwargs)


def shufflenet_g3_w1(**kwargs):
    """
    ShuffleNet 1x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=3, width_scale=1.0, model_name="shufflenet_g3_w1", **kwargs)


def shufflenet_g4_w1(**kwargs):
    """
    ShuffleNet 1x (g=4) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=4, width_scale=1.0, model_name="shufflenet_g4_w1", **kwargs)


def shufflenet_g8_w1(**kwargs):
    """
    ShuffleNet 1x (g=8) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=8, width_scale=1.0, model_name="shufflenet_g8_w1", **kwargs)


def shufflenet_g1_w3d4(**kwargs):
    """
    ShuffleNet 0.75x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=1, width_scale=0.75, model_name="shufflenet_g1_w3d4", **kwargs)


def shufflenet_g3_w3d4(**kwargs):
    """
    ShuffleNet 0.75x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=3, width_scale=0.75, model_name="shufflenet_g3_w3d4", **kwargs)


def shufflenet_g1_wd2(**kwargs):
    """
    ShuffleNet 0.5x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=1, width_scale=0.5, model_name="shufflenet_g1_wd2", **kwargs)


def shufflenet_g3_wd2(**kwargs):
    """
    ShuffleNet 0.5x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=3, width_scale=0.5, model_name="shufflenet_g3_wd2", **kwargs)


def shufflenet_g1_wd4(**kwargs):
    """
    ShuffleNet 0.25x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=1, width_scale=0.25, model_name="shufflenet_g1_wd4", **kwargs)


def shufflenet_g3_wd4(**kwargs):
    """
    ShuffleNet 0.25x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shufflenet(groups=3, width_scale=0.25, model_name="shufflenet_g3_wd4", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        shufflenet_g1_w1,
        shufflenet_g2_w1,
        shufflenet_g3_w1,
        shufflenet_g4_w1,
        shufflenet_g8_w1,
        shufflenet_g1_w3d4,
        shufflenet_g3_w3d4,
        shufflenet_g1_wd2,
        shufflenet_g3_wd2,
        shufflenet_g1_wd4,
        shufflenet_g3_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shufflenet_g1_w1 or weight_count == 1531936)
        assert (model != shufflenet_g2_w1 or weight_count == 1733848)
        assert (model != shufflenet_g3_w1 or weight_count == 1865728)
        assert (model != shufflenet_g4_w1 or weight_count == 1968344)
        assert (model != shufflenet_g8_w1 or weight_count == 2434768)
        assert (model != shufflenet_g1_w3d4 or weight_count == 975214)
        assert (model != shufflenet_g3_w3d4 or weight_count == 1238266)
        assert (model != shufflenet_g1_wd2 or weight_count == 534484)
        assert (model != shufflenet_g3_wd2 or weight_count == 718324)
        assert (model != shufflenet_g1_wd4 or weight_count == 209746)
        assert (model != shufflenet_g3_wd4 or weight_count == 305902)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
