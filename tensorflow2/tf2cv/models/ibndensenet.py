"""
    IBN-DenseNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.
"""

__all__ = ['IBNDenseNet', 'ibn_densenet121', 'ibn_densenet161', 'ibn_densenet169', 'ibn_densenet201']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, BatchNorm, pre_conv3x3_block, IBN, SimpleSequential, flatten, is_channels_first,\
    get_channel_axis
from .preresnet import PreResInitBlock, PreResActivation
from .densenet import TransitionBlock


class IBNPreConvBlock(nn.Layer):
    """
    IBN-Net specific convolution block with BN/IBN normalization and ReLU pre-activation.

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
    use_ibn : bool, default False
        Whether use Instance-Batch Normalization.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_ibn=False,
                 return_preact=False,
                 data_format="channels_last",
                 **kwargs):
        super(IBNPreConvBlock, self).__init__(**kwargs)
        self.use_ibn = use_ibn
        self.return_preact = return_preact

        if self.use_ibn:
            self.ibn = IBN(
                channels=in_channels,
                first_fraction=0.6,
                inst_first=False,
                data_format=data_format,
                name="ibn")
        else:
            self.bn = BatchNorm(
                data_format=data_format,
                name="bn")
        self.activ = nn.ReLU()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        if self.use_ibn:
            x = self.ibn(x, training=training)
        else:
            x = self.bn(x, training=training)
        x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x, training=training)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def ibn_pre_conv1x1_block(in_channels,
                          out_channels,
                          strides=1,
                          use_ibn=False,
                          return_preact=False,
                          data_format="channels_last",
                          **kwargs):
    """
    1x1 version of the IBN-Net specific pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_ibn : bool, default False
        Whether use Instance-Batch Normalization.
    return_preact : bool, default False
        Whether return pre-activation.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return IBNPreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        use_ibn=use_ibn,
        return_preact=return_preact,
        data_format=data_format,
        **kwargs)


class IBNDenseUnit(nn.Layer):
    """
    IBN-DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    conv1_ibn : bool
        Whether to use IBN normalization in the first convolution layer of the block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate,
                 conv1_ibn,
                 data_format="channels_last",
                 **kwargs):
        super(IBNDenseUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        self.conv1 = ibn_pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_ibn=conv1_ibn,
            data_format=data_format,
            name="conv1")
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=inc_channels,
            data_format=data_format,
            name="conv2")
        if self.use_dropout:
            self.dropout = nn.Dropout(
                rate=dropout_rate,
                name="dropout")

    def call(self, x, training=None):
        identity = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        if self.use_dropout:
            x = self.dropout(x, training=training)
        x = tf.concat([identity, x], axis=get_channel_axis(self.data_format))
        return x


class IBNDenseNet(tf.keras.Model):
    """
    IBN-DenseNet model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(IBNDenseNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(PreResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            if i != 0:
                stage.add(TransitionBlock(
                    in_channels=in_channels,
                    out_channels=(in_channels // 2),
                    data_format=data_format,
                    name="trans{}".format(i + 1)))
                in_channels = in_channels // 2
            for j, out_channels in enumerate(channels_per_stage):
                conv1_ibn = (i < 3) and (j % 3 == 0)
                stage.add(IBNDenseUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    conv1_ibn=conv1_ibn,
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


def get_ibndensenet(num_layers,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".tensorflow", "models"),
                    **kwargs):
    """
    Create IBN-DenseNet model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if num_layers == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif num_layers == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif num_layers == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif num_layers == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported IBN-DenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = IBNDenseNet(
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


def ibn_densenet121(**kwargs):
    """
    IBN-DenseNet-121 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=121, model_name="ibn_densenet121", **kwargs)


def ibn_densenet161(**kwargs):
    """
    IBN-DenseNet-161 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=161, model_name="ibn_densenet161", **kwargs)


def ibn_densenet169(**kwargs):
    """
    IBN-DenseNet-169 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=169, model_name="ibn_densenet169", **kwargs)


def ibn_densenet201(**kwargs):
    """
    IBN-DenseNet-201 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=201, model_name="ibn_densenet201", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        ibn_densenet121,
        ibn_densenet161,
        ibn_densenet169,
        ibn_densenet201,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibn_densenet121 or weight_count == 7978856)
        assert (model != ibn_densenet161 or weight_count == 28681000)
        assert (model != ibn_densenet169 or weight_count == 14149480)
        assert (model != ibn_densenet201 or weight_count == 20013928)


if __name__ == "__main__":
    _test()
