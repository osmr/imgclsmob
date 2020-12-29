"""
    Xception for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Xception: Deep Learning with Depthwise Separable Convolutions,' https://arxiv.org/abs/1610.02357.
"""

__all__ = ['Xception', 'xception']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, BatchNorm, MaxPool2d, AvgPool2d, conv1x1_block, conv3x3_block, flatten,\
    SimpleSequential, is_channels_first


class DwsConv(nn.Layer):
    """
    Depthwise separable convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 data_format="channels_last",
                 **kwargs):
        super(DwsConv, self).__init__(**kwargs)
        self.dw_conv = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=in_channels,
            use_bias=False,
            data_format=data_format,
            name="dw_conv")
        self.pw_conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name="pw_conv")

    def call(self, x, training=None):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DwsConvBlock(nn.Layer):
    """
    Depthwise separable convolution block with batchnorm and ReLU pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 activate,
                 data_format="channels_last",
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        self.activate = activate

        if self.activate:
            self.activ = nn.ReLU()
        self.conv = DwsConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name="conv")
        self.bn = BatchNorm(
            data_format=data_format,
            name="bn")

    def call(self, x, training=None):
        if self.activate:
            x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


def dws_conv3x3_block(in_channels,
                      out_channels,
                      activate,
                      data_format="channels_last",
                      **kwargs):
    """
    3x3 version of the depthwise separable convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1,
        activate=activate,
        data_format=data_format,
        **kwargs)


class XceptionUnit(nn.Layer):
    """
    Xception unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the downsample polling.
    reps : int
        Number of repetitions.
    start_with_relu : bool, default True
        Whether start with ReLU activation.
    grow_first : bool, default True
        Whether start from growing.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 reps,
                 start_with_relu=True,
                 grow_first=True,
                 data_format="channels_last",
                 **kwargs):
        super(XceptionUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activation=None,
                data_format=data_format,
                name="identity_conv")

        self.body = SimpleSequential(name="body")
        for i in range(reps):
            if (grow_first and (i == 0)) or ((not grow_first) and (i == reps - 1)):
                in_channels_i = in_channels
                out_channels_i = out_channels
            else:
                if grow_first:
                    in_channels_i = out_channels
                    out_channels_i = out_channels
                else:
                    in_channels_i = in_channels
                    out_channels_i = in_channels
            activate = start_with_relu if (i == 0) else True
            self.body.children.append(dws_conv3x3_block(
                in_channels=in_channels_i,
                out_channels=out_channels_i,
                activate=activate,
                data_format=data_format,
                name="block{}".format(i + 1)))
        if strides != 1:
            self.body.children.append(MaxPool2d(
                pool_size=3,
                strides=strides,
                padding=1,
                data_format=data_format,
                name="pool"))

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = tf.identity(x)
        x = self.body(x, training=training)
        x = x + identity
        return x


class XceptionInitBlock(nn.Layer):
    """
    Xception specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 data_format="channels_last",
                 **kwargs):
        super(XceptionInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            strides=2,
            padding=0,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=64,
            strides=1,
            padding=0,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class XceptionFinalBlock(nn.Layer):
    """
    Xception specific final block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(XceptionFinalBlock, self).__init__(**kwargs)
        self.conv1 = dws_conv3x3_block(
            in_channels=1024,
            out_channels=1536,
            activate=False,
            data_format=data_format,
            name="conv1")
        self.conv2 = dws_conv3x3_block(
            in_channels=1536,
            out_channels=2048,
            activate=True,
            data_format=data_format,
            name="conv2")
        self.activ = nn.ReLU()
        self.pool = AvgPool2d(
            pool_size=10,
            strides=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.activ(x)
        x = self.pool(x)
        return x


class Xception(tf.keras.Model):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
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
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(Xception, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(XceptionInitBlock(
            in_channels=in_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = 64
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                stage.add(XceptionUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=(2 if (j == 0) else 1),
                    reps=(2 if (j == 0) else 3),
                    start_with_relu=((i != 0) or (j != 0)),
                    grow_first=((i != len(channels) - 1) or (j != len(channels_per_stage) - 1)),
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(XceptionFinalBlock(
            data_format=data_format,
            name="final_block"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=2048,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_xception(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create Xception model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    channels = [[128], [256], [728] * 9, [1024]]

    net = Xception(
        channels=channels,
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


def xception(**kwargs):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_xception(model_name="xception", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        xception,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 299, 299) if is_channels_first(data_format) else (batch, 299, 299, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xception or weight_count == 22855952)


if __name__ == "__main__":
    _test()
