"""
    ResNeXt for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
"""

__all__ = ['ResNeXt', 'resnext14_16x4d', 'resnext14_32x2d', 'resnext14_32x4d', 'resnext26_16x4d', 'resnext26_32x2d',
           'resnext26_32x4d', 'resnext38_32x4d', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
           'ResNeXtBottleneck', 'ResNeXtUnit']

import os
import math
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, flatten
from .common import add_get_config
from .resnet import ResInitBlock


class ResNeXtBottleneck(nn.Layer):
    """
    ResNeXt bottleneck block for residual path in ResNeXt unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    @add_get_config
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 bottleneck_factor=4,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeXtBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=group_width,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=group_width,
            out_channels=group_width,
            strides=strides,
            groups=cardinality,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1_block(
            in_channels=group_width,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class ResNeXtUnit(nn.Layer):
    """
    ResNeXt unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    @add_get_config
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeXtUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = ResNeXtBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width,
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


class ResNeXt(tf.keras.Model):
    """
    ResNeXt model from 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    @add_get_config
    def __init__(self,
                 channels,
                 init_block_channels,
                 cardinality,
                 bottleneck_width,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeXt, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = tf.keras.Sequential(name="features")
        self.features.add(ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = tf.keras.Sequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(ResNeXtUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
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


def get_resnext(blocks,
                cardinality,
                bottleneck_width,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create ResNeXt model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if blocks == 14:
        layers = [1, 1, 1, 1]
    elif blocks == 26:
        layers = [2, 2, 2, 2]
    elif blocks == 38:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported ResNeXt with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ResNeXt(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
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


def resnext14_16x4d(**kwargs):
    """
    ResNeXt-14 (16x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=14, cardinality=16, bottleneck_width=4, model_name="resnext14_16x4d", **kwargs)


def resnext14_32x2d(**kwargs):
    """
    ResNeXt-14 (32x2d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=14, cardinality=32, bottleneck_width=2, model_name="resnext14_32x2d", **kwargs)


def resnext14_32x4d(**kwargs):
    """
    ResNeXt-14 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=14, cardinality=32, bottleneck_width=4, model_name="resnext14_32x4d", **kwargs)


def resnext26_16x4d(**kwargs):
    """
    ResNeXt-26 (16x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=26, cardinality=16, bottleneck_width=4, model_name="resnext26_16x4d", **kwargs)


def resnext26_32x2d(**kwargs):
    """
    ResNeXt-26 (32x2d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=26, cardinality=32, bottleneck_width=2, model_name="resnext26_32x2d", **kwargs)


def resnext26_32x4d(**kwargs):
    """
    ResNeXt-26 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=26, cardinality=32, bottleneck_width=4, model_name="resnext26_32x4d", **kwargs)


def resnext38_32x4d(**kwargs):
    """
    ResNeXt-38 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=38, cardinality=32, bottleneck_width=4, model_name="resnext38_32x4d", **kwargs)


def resnext50_32x4d(**kwargs):
    """
    ResNeXt-50 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="resnext50_32x4d", **kwargs)


def resnext101_32x4d(**kwargs):
    """
    ResNeXt-101 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="resnext101_32x4d", **kwargs)


def resnext101_64x4d(**kwargs):
    """
    ResNeXt-101 (64x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="resnext101_64x4d", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        resnext14_16x4d,
        resnext14_32x2d,
        resnext14_32x4d,
        resnext26_16x4d,
        resnext26_32x2d,
        resnext26_32x4d,
        resnext38_32x4d,
        resnext50_32x4d,
        resnext101_32x4d,
        resnext101_64x4d,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch_saze = 14
        x = tf.random.normal((batch_saze, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch_saze, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnext14_16x4d or weight_count == 7127336)
        assert (model != resnext14_32x2d or weight_count == 7029416)
        assert (model != resnext14_32x4d or weight_count == 9411880)
        assert (model != resnext26_16x4d or weight_count == 10119976)
        assert (model != resnext26_32x2d or weight_count == 9924136)
        assert (model != resnext26_32x4d or weight_count == 15389480)
        assert (model != resnext38_32x4d or weight_count == 21367080)
        assert (model != resnext50_32x4d or weight_count == 25028904)
        assert (model != resnext101_32x4d or weight_count == 44177704)
        assert (model != resnext101_64x4d or weight_count == 83455272)


if __name__ == "__main__":
    _test()
