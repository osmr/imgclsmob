"""
    DiracNetV2 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.
"""

__all__ = ['DiracNetV2', 'diracnet18v2', 'diracnet34v2']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, MaxPool2d, SimpleSequential, flatten, is_channels_first


class DiracConv(nn.Layer):
    """
    DiracNetV2 specific convolution block with pre-activation.

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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 data_format="channels_last",
                 **kwargs):
        super(DiracConv, self).__init__(**kwargs)
        self.activ = nn.ReLU()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.activ(x)
        x = self.conv(x)
        return x


def dirac_conv3x3(in_channels,
                  out_channels,
                  data_format="channels_last",
                  **kwargs):
    """
    3x3 version of the DiracNetV2 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DiracConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1,
        data_format=data_format,
        **kwargs)


class DiracInitBlock(nn.Layer):
    """
    DiracNetV2 specific initial block.

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
        super(DiracInitBlock, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            strides=2,
            padding=3,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DiracNetV2(tf.keras.Model):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
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
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DiracNetV2, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(DiracInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                stage.add(dirac_conv3x3(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            if i != len(channels) - 1:
                stage.add(MaxPool2d(
                    pool_size=2,
                    strides=2,
                    padding=0,
                    data_format=data_format,
                    name="pool{}".format(i + 1)))
            self.features.add(stage)
        self.features.add(nn.ReLU(name="final_activ"))
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


def get_diracnetv2(blocks,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".tensorflow", "models"),
                   **kwargs):
    """
    Create DiracNetV2 model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if blocks == 18:
        layers = [4, 4, 4, 4]
    elif blocks == 34:
        layers = [6, 8, 12, 6]
    else:
        raise ValueError("Unsupported DiracNetV2 with number of blocks: {}".format(blocks))

    channels_per_layers = [64, 128, 256, 512]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    init_block_channels = 64

    net = DiracNetV2(
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


def diracnet18v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=18, model_name="diracnet18v2", **kwargs)


def diracnet34v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=34, model_name="diracnet34v2", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        diracnet18v2,
        diracnet34v2,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diracnet18v2 or weight_count == 11511784)
        assert (model != diracnet34v2 or weight_count == 21616232)


if __name__ == "__main__":
    _test()
