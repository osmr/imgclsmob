"""
    SE-ResNeXt for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['SEResNeXt', 'seresnext50_32x4d', 'seresnext101_32x4d', 'seresnext101_64x4d']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, SEBlock, SimpleSequential, flatten
from .resnet import ResInitBlock
from .resnext import ResNeXtBottleneck


class SEResNeXtUnit(nn.Layer):
    """
    SE-ResNeXt unit.

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
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 data_format="channels_last",
                 **kwargs):
        super(SEResNeXtUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = ResNeXtBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width,
            data_format=data_format,
            name="body")
        self.se = SEBlock(
            channels=out_channels,
            data_format=data_format,
            name="se")
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
        x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class SEResNeXt(tf.keras.Model):
    """
    SE-ResNeXt model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
        super(SEResNeXt, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(SEResNeXtUnit(
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


def get_seresnext(blocks,
                  cardinality,
                  bottleneck_width,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
                  **kwargs):
    """
    Create SE-ResNeXt model with specific parameters.

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

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported SE-ResNeXt with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SEResNeXt(
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


def seresnext50_32x4d(**kwargs):
    """
    SE-ResNeXt-50 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="seresnext50_32x4d", **kwargs)


def seresnext101_32x4d(**kwargs):
    """
    SE-ResNeXt-101 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="seresnext101_32x4d", **kwargs)


def seresnext101_64x4d(**kwargs):
    """
    SE-ResNeXt-101 (64x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="seresnext101_64x4d", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        seresnext50_32x4d,
        seresnext101_32x4d,
        seresnext101_64x4d,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != seresnext50_32x4d or weight_count == 27559896)
        assert (model != seresnext101_32x4d or weight_count == 48955416)
        assert (model != seresnext101_64x4d or weight_count == 88232984)


if __name__ == "__main__":
    _test()
