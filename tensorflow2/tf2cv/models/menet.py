"""
    MENet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications,'
    https://arxiv.org/abs/1803.09127.
"""

__all__ = ['MENet', 'menet108_8x1_g3', 'menet128_8x1_g4', 'menet160_8x1_g8', 'menet228_12x1_g3', 'menet256_12x1_g4',
           'menet348_12x1_g3', 'menet352_12x1_g8', 'menet456_24x1_g3']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv3x3, depthwise_conv3x3, ChannelShuffle, Conv2d, BatchNorm, AvgPool2d,\
    MaxPool2d, SimpleSequential, get_channel_axis, flatten


class MEUnit(nn.Layer):
    """
    MENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    side_channels : int
        Number of side channels.
    groups : int
        Number of groups in convolution layers.
    downsample : bool
        Whether do downsample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 side_channels,
                 groups,
                 downsample,
                 ignore_group,
                 data_format="channels_last",
                 **kwargs):
        super(MEUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        # residual branch
        self.compress_conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=(1 if ignore_group else groups),
            data_format=data_format,
            name="compress_conv1")
        self.compress_bn1 = BatchNorm(
            # in_channels=mid_channels,
            data_format=data_format,
            name="compress_bn1")
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=groups,
            data_format=data_format,
            name="c_shuffle")
        self.dw_conv2 = depthwise_conv3x3(
            channels=mid_channels,
            strides=(2 if self.downsample else 1),
            data_format=data_format,
            name="dw_conv2")
        self.dw_bn2 = BatchNorm(
            # in_channels=mid_channels,
            data_format=data_format,
            name="dw_bn2")
        self.expand_conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            groups=groups,
            data_format=data_format,
            name="expand_conv3")
        self.expand_bn3 = BatchNorm(
            # in_channels=out_channels,
            data_format=data_format,
            name="expand_bn3")
        if downsample:
            self.avgpool = AvgPool2d(
                pool_size=3,
                strides=2,
                padding=1,
                data_format=data_format,
                name="avgpool")
        self.activ = nn.ReLU()

        # fusion branch
        self.s_merge_conv = conv1x1(
            in_channels=mid_channels,
            out_channels=side_channels,
            data_format=data_format,
            name="s_merge_conv")
        self.s_merge_bn = BatchNorm(
            # in_channels=side_channels,
            data_format=data_format,
            name="s_merge_bn")
        self.s_conv = conv3x3(
            in_channels=side_channels,
            out_channels=side_channels,
            strides=(2 if self.downsample else 1),
            data_format=data_format,
            name="s_conv")
        self.s_conv_bn = BatchNorm(
            # in_channels=side_channels,
            data_format=data_format,
            name="s_conv_bn")
        self.s_evolve_conv = conv1x1(
            in_channels=side_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="s_evolve_conv")
        self.s_evolve_bn = BatchNorm(
            # in_channels=mid_channels,
            data_format=data_format,
            name="s_evolve_bn")

    def call(self, x, training=None):
        identity = x
        # pointwise group convolution 1
        x = self.compress_conv1(x)
        x = self.compress_bn1(x, training=training)
        x = self.activ(x)
        x = self.c_shuffle(x)
        # merging
        y = self.s_merge_conv(x)
        y = self.s_merge_bn(y, training=training)
        y = self.activ(y)
        # depthwise convolution (bottleneck)
        x = self.dw_conv2(x)
        x = self.dw_bn2(x, training=training)
        # evolution
        y = self.s_conv(y)
        y = self.s_conv_bn(y, training=training)
        y = self.activ(y)
        y = self.s_evolve_conv(y)
        y = self.s_evolve_bn(y, training=training)
        y = tf.nn.sigmoid(y)
        x = x * y
        # pointwise group convolution 2
        x = self.expand_conv3(x)
        x = self.expand_bn3(x, training=training)
        # identity branch
        if self.downsample:
            identity = self.avgpool(identity)
            x = tf.concat([x, identity], axis=get_channel_axis(self.data_format))
        else:
            x = x + identity
        x = self.activ(x)
        return x


class MEInitBlock(nn.Layer):
    """
    MENet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(MEInitBlock, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=2,
            padding=1,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = BatchNorm(
            # in_channels=out_channels,
            data_format=data_format,
            name="bn")
        self.activ = nn.ReLU()
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activ(x)
        x = self.pool(x)
        return x


class MENet(tf.keras.Model):
    """
    MENet model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications,'
    https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    side_channels : int
        Number of side channels in a ME-unit.
    groups : int
        Number of groups in convolution layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 side_channels,
                 groups,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(MENet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(MEInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                ignore_group = (i == 0) and (j == 0)
                stage.add(MEUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    side_channels=side_channels,
                    groups=groups,
                    downsample=downsample,
                    ignore_group=ignore_group,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_menet(first_stage_channels,
              side_channels,
              groups,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create MENet model with specific parameters.

    Parameters:
    ----------
    first_stage_channels : int
        Number of output channels at the first stage.
    side_channels : int
        Number of side channels in a ME-unit.
    groups : int
        Number of groups in convolution layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    layers = [4, 8, 4]

    if first_stage_channels == 108:
        init_block_channels = 12
        channels_per_layers = [108, 216, 432]
    elif first_stage_channels == 128:
        init_block_channels = 12
        channels_per_layers = [128, 256, 512]
    elif first_stage_channels == 160:
        init_block_channels = 16
        channels_per_layers = [160, 320, 640]
    elif first_stage_channels == 228:
        init_block_channels = 24
        channels_per_layers = [228, 456, 912]
    elif first_stage_channels == 256:
        init_block_channels = 24
        channels_per_layers = [256, 512, 1024]
    elif first_stage_channels == 348:
        init_block_channels = 24
        channels_per_layers = [348, 696, 1392]
    elif first_stage_channels == 352:
        init_block_channels = 24
        channels_per_layers = [352, 704, 1408]
    elif first_stage_channels == 456:
        init_block_channels = 48
        channels_per_layers = [456, 912, 1824]
    else:
        raise ValueError("The {} of `first_stage_channels` is not supported".format(first_stage_channels))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = MENet(
        channels=channels,
        init_block_channels=init_block_channels,
        side_channels=side_channels,
        groups=groups,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def menet108_8x1_g3(**kwargs):
    """
    108-MENet-8x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=108, side_channels=8, groups=3, model_name="menet108_8x1_g3", **kwargs)


def menet128_8x1_g4(**kwargs):
    """
    128-MENet-8x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=128, side_channels=8, groups=4, model_name="menet128_8x1_g4", **kwargs)


def menet160_8x1_g8(**kwargs):
    """
    160-MENet-8x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=160, side_channels=8, groups=8, model_name="menet160_8x1_g8", **kwargs)


def menet228_12x1_g3(**kwargs):
    """
    228-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=228, side_channels=12, groups=3, model_name="menet228_12x1_g3", **kwargs)


def menet256_12x1_g4(**kwargs):
    """
    256-MENet-12x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=256, side_channels=12, groups=4, model_name="menet256_12x1_g4", **kwargs)


def menet348_12x1_g3(**kwargs):
    """
    348-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=348, side_channels=12, groups=3, model_name="menet348_12x1_g3", **kwargs)


def menet352_12x1_g8(**kwargs):
    """
    352-MENet-12x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=352, side_channels=12, groups=8, model_name="menet352_12x1_g8", **kwargs)


def menet456_24x1_g3(**kwargs):
    """
    456-MENet-24x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=456, side_channels=24, groups=3, model_name="menet456_24x1_g3", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        menet108_8x1_g3,
        menet128_8x1_g4,
        menet160_8x1_g8,
        menet228_12x1_g3,
        menet256_12x1_g4,
        menet348_12x1_g3,
        menet352_12x1_g8,
        menet456_24x1_g3,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != menet108_8x1_g3 or weight_count == 654516)
        assert (model != menet128_8x1_g4 or weight_count == 750796)
        assert (model != menet160_8x1_g8 or weight_count == 850120)
        assert (model != menet228_12x1_g3 or weight_count == 1806568)
        assert (model != menet256_12x1_g4 or weight_count == 1888240)
        assert (model != menet348_12x1_g3 or weight_count == 3368128)
        assert (model != menet352_12x1_g8 or weight_count == 2272872)
        assert (model != menet456_24x1_g3 or weight_count == 5304784)


if __name__ == "__main__":
    _test()
