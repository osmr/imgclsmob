"""
    Single-Path NASNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.
"""

__all__ = ['SPNASNet', 'spnasnet']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block, SimpleSequential, flatten,\
    is_channels_first


class SPNASUnit(nn.Layer):
    """
    Single-Path NASNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int
        Expansion factor for each unit.
    use_skip : bool, default True
        Whether to use skip connection.
    activation : str, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_kernel3,
                 exp_factor,
                 use_skip=True,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(SPNASUnit, self).__init__(**kwargs)
        assert (exp_factor >= 1)
        self.residual = (in_channels == out_channels) and (strides == 1) and use_skip
        self.use_exp_conv = exp_factor > 1
        mid_channels = exp_factor * in_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation,
                data_format=data_format,
                name="exp_conv")
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                activation=activation,
                data_format=data_format,
                name="conv1")
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                activation=activation,
                data_format=data_format,
                name="conv1")
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        if self.residual:
            x = x + identity
        return x


class SPNASInitBlock(nn.Layer):
    """
    Single-Path NASNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 data_format="channels_last",
                 **kwargs):
        super(SPNASInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv1")
        self.conv2 = SPNASUnit(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=1,
            use_kernel3=True,
            exp_factor=1,
            use_skip=False,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class SPNASFinalBlock(nn.Layer):
    """
    Single-Path NASNet specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 data_format="channels_last",
                 **kwargs):
        super(SPNASFinalBlock, self).__init__(**kwargs)
        self.conv1 = SPNASUnit(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=1,
            use_kernel3=True,
            exp_factor=6,
            use_skip=False,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class SPNASNet(tf.keras.Model):
    """
    Single-Path NASNet model from 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Number of output channels for the initial unit.
    final_block_channels : list of 2 int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
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
                 final_block_channels,
                 kernels3,
                 exp_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(SPNASNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(SPNASInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels[1],
            mid_channels=init_block_channels[0],
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels[1]
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if ((j == 0) and (i != 3)) or \
                               ((j == len(channels_per_stage) // 2) and (i == 3)) else 1
                use_kernel3 = kernels3[i][j] == 1
                exp_factor = exp_factors[i][j]
                stage.add(SPNASUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(SPNASFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels[1],
            mid_channels=final_block_channels[0],
            data_format=data_format,
            name="final_block"))
        in_channels = final_block_channels[1]
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


def get_spnasnet(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create Single-Path NASNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = [32, 16]
    final_block_channels = [320, 1280]
    channels = [[24, 24, 24], [40, 40, 40, 40], [80, 80, 80, 80], [96, 96, 96, 96, 192, 192, 192, 192]]
    kernels3 = [[1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]
    exp_factors = [[3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3, 6, 6, 6, 6]]

    net = SPNASNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
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


def spnasnet(**kwargs):
    """
    Single-Path NASNet model from 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_spnasnet(model_name="spnasnet", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        spnasnet,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != spnasnet or weight_count == 4421616)


if __name__ == "__main__":
    _test()
