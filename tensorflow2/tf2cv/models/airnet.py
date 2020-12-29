"""
    AirNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Attention Inspiring Receptive-Fields Network for Learning Invariant Representations,'
    https://ieeexplore.ieee.org/document/8510896.
"""

__all__ = ['AirNet', 'airnet50_1x64d_r2', 'airnet50_1x64d_r16', 'airnet101_1x64d_r2', 'AirBlock', 'AirInitBlock']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, MaxPool2d, SimpleSequential, flatten, is_channels_first


class AirBlock(nn.Layer):
    """
    AirNet attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int, default 1
        Number of groups.
    ratio: int, default 2
        Air compression ratio.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 ratio=2,
                 data_format="channels_last",
                 **kwargs):
        super(AirBlock, self).__init__(**kwargs)
        assert (out_channels % ratio == 0)
        mid_channels = out_channels // ratio

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            groups=groups,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv3")
        self.sigmoid = tf.nn.sigmoid
        self.upsample = nn.UpSampling2D(
            size=(2, 2),
            data_format=data_format,
            interpolation="bilinear",
            name="upsample")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.pool(x)
        x = self.conv2(x, training=training)
        x = self.upsample(x)
        x = self.conv3(x, training=training)
        x = self.sigmoid(x)
        return x


class AirBottleneck(nn.Layer):
    """
    AirNet bottleneck block for residual path in AirNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    ratio: int
        Air compression ratio.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 ratio,
                 data_format="channels_last",
                 **kwargs):
        super(AirBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4
        self.use_air_block = (strides == 1 and mid_channels < 512)

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv3")
        if self.use_air_block:
            self.air = AirBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                ratio=ratio,
                data_format=data_format,
                name="air")

    def call(self, x, training=None):
        if self.use_air_block:
            att = self.air(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        if self.use_air_block:
            x = x * att
        x = self.conv3(x, training=training)
        return x


class AirUnit(nn.Layer):
    """
    AirNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    ratio: int
        Air compression ratio.
    in_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 ratio,
                 data_format="channels_last",
                 **kwargs):
        super(AirUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = AirBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            ratio=ratio,
            data_format=data_format,
            name="body")
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activation=None,
                data_format=data_format,
                name="identity_conv")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class AirInitBlock(nn.Layer):
    """
    AirNet specific initial block.

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
        super(AirInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv3")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool(x)
        return x


class AirNet(tf.keras.Model):
    """
    AirNet model from 'Attention Inspiring Receptive-Fields Network for Learning Invariant Representations,'
    https://ieeexplore.ieee.org/document/8510896.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    ratio: int
        Air compression ratio.
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
                 ratio,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(AirNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(AirInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(AirUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    ratio=ratio,
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


def get_airnet(blocks,
               base_channels,
               ratio,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
               **kwargs):
    """
    Create AirNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    base_channels: int
        Base number of channels.
    ratio: int
        Air compression ratio.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported AirNet with number of blocks: {}".format(blocks))

    bottleneck_expansion = 4
    init_block_channels = base_channels
    channels_per_layers = [base_channels * (2 ** i) * bottleneck_expansion for i in range(len(layers))]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = AirNet(
        channels=channels,
        init_block_channels=init_block_channels,
        ratio=ratio,
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


def airnet50_1x64d_r2(**kwargs):
    """
    AirNet50-1x64d (r=2) model from 'Attention Inspiring Receptive-Fields Network for Learning Invariant
    Representations,' https://ieeexplore.ieee.org/document/8510896.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_airnet(blocks=50, base_channels=64, ratio=2, model_name="airnet50_1x64d_r2", **kwargs)


def airnet50_1x64d_r16(**kwargs):
    """
    AirNet50-1x64d (r=16) model from 'Attention Inspiring Receptive-Fields Network for Learning Invariant
    Representations,' https://ieeexplore.ieee.org/document/8510896.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_airnet(blocks=50, base_channels=64, ratio=16, model_name="airnet50_1x64d_r16", **kwargs)


def airnet101_1x64d_r2(**kwargs):
    """
    AirNet101-1x64d (r=2) model from 'Attention Inspiring Receptive-Fields Network for Learning Invariant
    Representations,' https://ieeexplore.ieee.org/document/8510896.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_airnet(blocks=101, base_channels=64, ratio=2, model_name="airnet101_1x64d_r2", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        airnet50_1x64d_r2,
        airnet50_1x64d_r16,
        airnet101_1x64d_r2,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != airnet50_1x64d_r2 or weight_count == 27425864)
        assert (model != airnet50_1x64d_r16 or weight_count == 25714952)
        assert (model != airnet101_1x64d_r2 or weight_count == 51727432)


if __name__ == "__main__":
    _test()
