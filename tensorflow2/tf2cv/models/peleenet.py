"""
    PeleeNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Pelee: A Real-Time Object Detection System on Mobile Devices,' https://arxiv.org/abs/1804.06882.
"""

__all__ = ['PeleeNet', 'peleenet']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, Concurrent, MaxPool2d, AvgPool2d, SimpleSequential, flatten,\
    is_channels_first, get_channel_axis


class PeleeBranch1(nn.Layer):
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
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 strides=1,
                 data_format="channels_last",
                 **kwargs):
        super(PeleeBranch1, self).__init__(**kwargs)
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=strides,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class PeleeBranch2(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 data_format="channels_last",
                 **kwargs):
        super(PeleeBranch2, self).__init__(**kwargs)
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class StemBlock(nn.Layer):
    """
    PeleeNet stem block.

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
        super(StemBlock, self).__init__(**kwargs)
        mid1_channels = out_channels // 2
        mid2_channels = out_channels * 2

        self.first_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            data_format=data_format,
            name="first_conv")

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(PeleeBranch1(
            in_channels=out_channels,
            out_channels=out_channels,
            mid_channels=mid1_channels,
            strides=2,
            data_format=data_format,
            name="branch1"))
        self.branches.add(MaxPool2d(
            pool_size=2,
            strides=2,
            padding=0,
            data_format=data_format,
            name="branch2"))

        self.last_conv = conv1x1_block(
            in_channels=mid2_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="last_conv")

    def call(self, x, training=None):
        x = self.first_conv(x, training=training)
        x = self.branches(x, training=training)
        x = self.last_conv(x, training=training)
        return x


class DenseBlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bottleneck_size,
                 data_format="channels_last",
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.data_format = data_format
        inc_channels = (out_channels - in_channels) // 2
        mid_channels = inc_channels * bottleneck_size

        self.branch1 = PeleeBranch1(
            in_channels=in_channels,
            out_channels=inc_channels,
            mid_channels=mid_channels,
            data_format=data_format,
            name="branch1")
        self.branch2 = PeleeBranch2(
            in_channels=in_channels,
            out_channels=inc_channels,
            mid_channels=mid_channels,
            data_format=data_format,
            name="branch2")

    def call(self, x, training=None):
        x1 = self.branch1(x, training=training)
        x2 = self.branch2(x, training=training)
        x = tf.concat([x, x1, x2], axis=get_channel_axis(self.data_format))
        return x


class TransitionBlock(nn.Layer):
    """
    PeleeNet's transition block, like in DensNet, but with ordinary convolution block.

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
        super(TransitionBlock, self).__init__(**kwargs)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")
        self.pool = AvgPool2d(
            pool_size=2,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x


class PeleeNet(tf.keras.Model):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck_sizes,
                 dropout_rate=0.5,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(PeleeNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(StemBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            bottleneck_size = bottleneck_sizes[i]
            stage = SimpleSequential(name="stage{}".format(i + 1))
            if i != 0:
                stage.add(TransitionBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    data_format=data_format,
                    name="trans{}".format(i + 1)))
            for j, out_channels in enumerate(channels_per_stage):
                stage.add(DenseBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bottleneck_size=bottleneck_size,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels,
            data_format=data_format,
            name="final_block"))
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        if dropout_rate > 0.0:
            self.output1.add(nn.Dropout(
                rate=dropout_rate,
                name="dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_peleenet(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create PeleeNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def peleenet(**kwargs):
    """
    PeleeNet model from 'Pelee: A Real-Time Object Detection System on Mobile Devices,'
    https://arxiv.org/abs/1804.06882.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_peleenet(model_name="peleenet", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        peleenet,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != peleenet or weight_count == 2802248)


if __name__ == "__main__":
    _test()
