"""
    PyramidNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.
"""

__all__ = ['PyramidNet', 'pyramidnet101_a360', 'PyrUnit']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, BatchNorm, MaxPool2d, AvgPool2d, pre_conv1x1_block, pre_conv3x3_block, SimpleSequential,\
    flatten, is_channels_first
from .preresnet import PreResActivation


class PyrBlock(nn.Layer):
    """
    Simple PyramidNet block for residual path in PyramidNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 data_format="channels_last",
                 **kwargs):
        super(PyrBlock, self).__init__(**kwargs)
        self.conv1 = pre_conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            activate=False,
            data_format=data_format,
            name="conv1")
        self.conv2 = pre_conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class PyrBottleneck(nn.Layer):
    """
    PyramidNet bottleneck block for residual path in PyramidNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 data_format="channels_last",
                 **kwargs):
        super(PyrBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            activate=False,
            data_format=data_format,
            name="conv1")
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            data_format=data_format,
            name="conv2")
        self.conv3 = pre_conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class PyrUnit(nn.Layer):
    """
    PyramidNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck,
                 data_format="channels_last",
                 **kwargs):
        super(PyrUnit, self).__init__(**kwargs)
        assert (out_channels >= in_channels)
        self.data_format = data_format
        self.resize_identity = (strides != 1)
        self.identity_pad_width = out_channels - in_channels

        if bottleneck:
            self.body = PyrBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                data_format=data_format,
                name="body")
        else:
            self.body = PyrBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                data_format=data_format,
                name="body")
        self.bn = BatchNorm(
            data_format=data_format,
            name="bn")
        if self.resize_identity:
            self.identity_pool = AvgPool2d(
                pool_size=2,
                strides=strides,
                ceil_mode=True,
                data_format=data_format,
                name="identity_pool")

    def call(self, x, training=None):
        identity = x
        x = self.body(x, training=training)
        x = self.bn(x, training=training)
        if self.resize_identity:
            identity = self.identity_pool(identity)
        if self.identity_pad_width > 0:
            if is_channels_first(self.data_format):
                paddings = [[0, 0], [0, self.identity_pad_width], [0, 0], [0, 0]]
            else:
                paddings = [[0, 0], [0, 0], [0, 0], [0, self.identity_pad_width]]
            identity = tf.pad(identity, paddings=paddings)
        x = x + identity
        return x


class PyrInitBlock(nn.Layer):
    """
    PyramidNet specific initial block.

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
        super(PyrInitBlock, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            strides=2,
            padding=3,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = BatchNorm(
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


class PyramidNet(tf.keras.Model):
    """
    PyramidNet model from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
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
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(PyramidNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(PyrInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(PyrUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bottleneck=bottleneck,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(PreResActivation(
            in_channels=in_channels,
            data_format=data_format,
            name="post_activ"))
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


def get_pyramidnet(blocks,
                   alpha,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".tensorflow", "models"),
                   **kwargs):
    """
    Create PyramidNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    alpha : int
        PyramidNet's alpha value.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14:
        layers = [2, 2, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    growth_add = float(alpha) / float(sum(layers))
    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [[(i + 1) * growth_add + xi[-1][-1] for i in list(range(yi))]],
        layers,
        [[init_block_channels]])[1:]
    channels = [[int(round(cij)) for cij in ci] for ci in channels]

    if blocks < 50:
        bottleneck = False
    else:
        bottleneck = True
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = PyramidNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
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


def pyramidnet101_a360(**kwargs):
    """
    PyramidNet-101 model from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet(blocks=101, alpha=360, model_name="pyramidnet101_a360", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        pyramidnet101_a360,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pyramidnet101_a360 or weight_count == 42455070)


if __name__ == "__main__":
    _test()
