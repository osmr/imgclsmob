"""
    SelecSLS for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.
"""

__all__ = ['SelecSLS', 'selecsls42', 'selecsls42b', 'selecsls60', 'selecsls60b', 'selecsls84']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, DualPathSequential, AvgPool2d, SimpleSequential, flatten,\
    is_channels_first, get_channel_axis


class SelecSLSBlock(nn.Layer):
    """
    SelecSLS block.

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
        super(SelecSLSBlock, self).__init__(**kwargs)
        mid_channels = 2 * out_channels

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

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class SelecSLSUnit(nn.Layer):
    """
    SelecSLS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    skip_channels : int
        Number of skipped channels.
    mid_channels : int
        Number of middle channels.
    strides : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_channels,
                 mid_channels,
                 strides,
                 data_format="channels_last",
                 **kwargs):
        super(SelecSLSUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.resize = (strides == 2)
        mid2_channels = mid_channels // 2
        last_channels = 2 * mid_channels + (skip_channels if strides == 1 else 0)

        self.branch1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=strides,
            data_format=data_format,
            name="branch1")
        self.branch2 = SelecSLSBlock(
            in_channels=mid_channels,
            out_channels=mid2_channels,
            data_format=data_format,
            name="branch2")
        self.branch3 = SelecSLSBlock(
            in_channels=mid2_channels,
            out_channels=mid2_channels,
            data_format=data_format,
            name="branch3")
        self.last_conv = conv1x1_block(
            in_channels=last_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="last_conv")

    def call(self, x, x0=None, training=None):
        x1 = self.branch1(x, training=training)
        x2 = self.branch2(x1, training=training)
        x3 = self.branch3(x2, training=training)
        if self.resize:
            y = tf.concat([x1, x2, x3], axis=get_channel_axis(self.data_format))
            y = self.last_conv(y, training=training)
            return y, y
        else:
            y = tf.concat([x1, x2, x3, x0], axis=get_channel_axis(self.data_format))
            y = self.last_conv(y, training=training)
            return y, x0


class SelecSLS(tf.keras.Model):
    """
    SelecSLS model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    skip_channels : list of list of int
        Number of skipped channels for each unit.
    mid_channels : list of list of int
        Number of middle channels for each unit.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 1x1) kernel for each head unit.
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
                 skip_channels,
                 mid_channels,
                 kernels3,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(SelecSLS, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        init_block_channels = 32

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=(1 + len(kernels3)),
            name="features")
        self.features.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            strides=2,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            k = i - len(skip_channels)
            stage = DualPathSequential(name="stage{}".format(i + 1)) if k < 0 else\
                SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if j == 0 else 1
                if k < 0:
                    unit = SelecSLSUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        skip_channels=skip_channels[i][j],
                        mid_channels=mid_channels[i][j],
                        strides=strides,
                        data_format=data_format,
                        name="unit{}".format(j + 1))
                else:
                    conv_block_class = conv3x3_block if kernels3[k][j] == 1 else conv1x1_block
                    unit = conv_block_class(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=strides,
                        data_format=data_format,
                        name="unit{}".format(j + 1))
                stage.add(unit)
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(AvgPool2d(
            pool_size=4,
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


def get_selecsls(version,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create SelecSLS model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SelecSLS.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if version in ("42", "42b"):
        channels = [[64, 128], [144, 288], [304, 480]]
        skip_channels = [[0, 64], [0, 144], [0, 304]]
        mid_channels = [[64, 64], [144, 144], [304, 304]]
        kernels3 = [[1, 1], [1, 0]]
        if version == "42":
            head_channels = [[960, 1024], [1024, 1280]]
        else:
            head_channels = [[960, 1024], [1280, 1024]]
    elif version in ("60", "60b"):
        channels = [[64, 128], [128, 128, 288], [288, 288, 288, 416]]
        skip_channels = [[0, 64], [0, 128, 128], [0, 288, 288, 288]]
        mid_channels = [[64, 64], [128, 128, 128], [288, 288, 288, 288]]
        kernels3 = [[1, 1], [1, 0]]
        if version == "60":
            head_channels = [[756, 1024], [1024, 1280]]
        else:
            head_channels = [[756, 1024], [1280, 1024]]
    elif version == "84":
        channels = [[64, 144], [144, 144, 144, 144, 304], [304, 304, 304, 304, 304, 512]]
        skip_channels = [[0, 64], [0, 144, 144, 144, 144], [0, 304, 304, 304, 304, 304]]
        mid_channels = [[64, 64], [144, 144, 144, 144, 144], [304, 304, 304, 304, 304, 304]]
        kernels3 = [[1, 1], [1, 1]]
        head_channels = [[960, 1024], [1024, 1280]]
    else:
        raise ValueError("Unsupported SelecSLS version {}".format(version))

    channels += head_channels

    net = SelecSLS(
        channels=channels,
        skip_channels=skip_channels,
        mid_channels=mid_channels,
        kernels3=kernels3,
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


def selecsls42(**kwargs):
    """
    SelecSLS-42 model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="42", model_name="selecsls42", **kwargs)


def selecsls42b(**kwargs):
    """
    SelecSLS-42b model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="42b", model_name="selecsls42b", **kwargs)


def selecsls60(**kwargs):
    """
    SelecSLS-60 model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="60", model_name="selecsls60", **kwargs)


def selecsls60b(**kwargs):
    """
    SelecSLS-60b model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="60b", model_name="selecsls60b", **kwargs)


def selecsls84(**kwargs):
    """
    SelecSLS-84 model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="84", model_name="selecsls84", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        selecsls42,
        selecsls42b,
        selecsls60,
        selecsls60b,
        selecsls84,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != selecsls42 or weight_count == 30354952)
        assert (model != selecsls42b or weight_count == 32458248)
        assert (model != selecsls60 or weight_count == 30670768)
        assert (model != selecsls60b or weight_count == 32774064)
        assert (model != selecsls84 or weight_count == 50954600)


if __name__ == "__main__":
    _test()
