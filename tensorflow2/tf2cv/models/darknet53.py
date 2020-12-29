"""
    DarkNet-53 for ImageNet-1K, implemented in TensorFlow.
    Original source: 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.
"""

__all__ = ['DarkNet53', 'darknet53']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, SimpleSequential, flatten


class DarkUnit(nn.Layer):
    """
    DarkNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    alpha : float
        Slope coefficient for Leaky ReLU activation.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha,
                 data_format="channels_last",
                 **kwargs):
        super(DarkUnit, self).__init__(**kwargs)
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=nn.LeakyReLU(alpha=alpha),
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=nn.LeakyReLU(alpha=alpha),
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        identity = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x + identity


class DarkNet53(tf.keras.Model):
    """
    DarkNet-53 model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
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
                 alpha=0.1,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DarkNet53, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            activation=nn.LeakyReLU(alpha=alpha),
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    stage.add(conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=2,
                        activation=nn.LeakyReLU(alpha=alpha),
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                else:
                    stage.add(DarkUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        alpha=alpha,
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


def get_darknet53(model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
                  **kwargs):
    """
    Create DarkNet model with specific parameters.

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
    layers = [2, 3, 9, 9, 5]
    channels_per_layers = [64, 128, 256, 512, 1024]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = DarkNet53(
        channels=channels,
        init_block_channels=init_block_channels,
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


def darknet53(**kwargs):
    """
    DarkNet-53 'Reference' model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_darknet53(model_name="darknet53", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        darknet53,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darknet53 or weight_count == 41609928)


if __name__ == "__main__":
    _test()
