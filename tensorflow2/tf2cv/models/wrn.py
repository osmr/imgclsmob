"""
    WRN for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['WRN', 'wrn50_2']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, MaxPool2d, SimpleSequential, flatten, is_channels_first


class WRNConv(nn.Layer):
    """
    WRN specific convolution block.

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
        super(WRNConv, self).__init__(**kwargs)
        self.activate = activate

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            data_format=data_format,
            name="conv")
        if self.activate:
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        x = self.conv(x)
        if self.activate:
            x = self.activ(x)
        return x


def wrn_conv1x1(in_channels,
                out_channels,
                strides,
                activate,
                data_format="channels_last",
                **kwargs):
    """
    1x1 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        activate=activate,
        data_format=data_format,
        **kwargs)


def wrn_conv3x3(in_channels,
                out_channels,
                strides,
                activate,
                data_format="channels_last",
                **kwargs):
    """
    3x3 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        activate=activate,
        data_format=data_format,
        **kwargs)


class WRNBottleneck(nn.Layer):
    """
    WRN bottleneck block for residual path in WRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 width_factor,
                 data_format="channels_last",
                 **kwargs):
        super(WRNBottleneck, self).__init__(**kwargs)
        mid_channels = int(round(out_channels // 4 * width_factor))

        self.conv1 = wrn_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=1,
            activate=True,
            data_format=data_format,
            name="conv1")
        self.conv2 = wrn_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            activate=True,
            data_format=data_format,
            name="conv2")
        self.conv3 = wrn_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=1,
            activate=False,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class WRNUnit(nn.Layer):
    """
    WRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 width_factor,
                 data_format="channels_last",
                 **kwargs):
        super(WRNUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = WRNBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            width_factor=width_factor,
            data_format=data_format,
            name="body")
        if self.resize_identity:
            self.identity_conv = wrn_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activate=False,
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


class WRNInitBlock(nn.Layer):
    """
    WRN specific initial block.

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
        super(WRNInitBlock, self).__init__(**kwargs)
        self.conv = WRNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            strides=2,
            padding=3,
            activate=True,
            data_format=data_format,
            name="conv")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x


class WRN(tf.keras.Model):
    """
    WRN model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    width_factor : float
        Wide scale factor for width of layers.
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
                 width_factor,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(WRN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(WRNInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(WRNUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    width_factor=width_factor,
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


def get_wrn(blocks,
            width_factor,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".tensorflow", "models"),
            **kwargs):
    """
    Create WRN model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    width_factor : float
        Wide scale factor for width of layers.
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
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported WRN with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = WRN(
        channels=channels,
        init_block_channels=init_block_channels,
        width_factor=width_factor,
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


def wrn50_2(**kwargs):
    """
    WRN-50-2 model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn(blocks=50, width_factor=2.0, model_name="wrn50_2", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        wrn50_2,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn50_2 or weight_count == 68849128)


if __name__ == "__main__":
    _test()
