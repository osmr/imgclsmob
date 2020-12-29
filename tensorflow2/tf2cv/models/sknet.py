"""
    SKNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.
"""

__all__ = ['SKNet', 'sknet50', 'sknet101', 'sknet152']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent, SimpleSequential, flatten, is_channels_first,\
    get_channel_axis
from .resnet import ResInitBlock


class SKConvBlock(nn.Layer):
    """
    SKNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int, default 32
        Number of groups in branches.
    num_branches : int, default 2
        Number of branches (`M` parameter in the paper).
    reduction : int, default 16
        Reduction value for intermediate channels (`r` parameter in the paper).
    min_channels : int, default 32
        Minimal number of intermediate channels (`L` parameter in the paper).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 groups=32,
                 num_branches=2,
                 reduction=16,
                 min_channels=32,
                 data_format="channels_last",
                 **kwargs):
        super(SKConvBlock, self).__init__(**kwargs)
        self.num_branches = num_branches
        self.out_channels = out_channels
        self.data_format = data_format
        self.axis = get_channel_axis(data_format)
        mid_channels = max(in_channels // reduction, min_channels)

        self.branches = Concurrent(
            stack=True,
            data_format=data_format,
            name="branches")
        for i in range(num_branches):
            dilation = 1 + i
            self.branches.children.append(conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                data_format=data_format,
                name="branch{}".format(i + 2)))
        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.fc1 = conv1x1_block(
            in_channels=out_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="fc1")
        self.fc2 = conv1x1(
            in_channels=mid_channels,
            out_channels=(out_channels * num_branches),
            data_format=data_format,
            name="fc2")
        self.softmax = nn.Softmax(axis=self.axis)

    def call(self, x, training=None):
        y = self.branches(x)

        u = tf.math.reduce_sum(y, axis=self.axis)
        s = self.pool(u)
        if is_channels_first(self.data_format):
            s = tf.expand_dims(tf.expand_dims(s, 2), 3)
        else:
            s = tf.expand_dims(tf.expand_dims(s, 1), 2)
        z = self.fc1(s)
        w = self.fc2(z)

        if is_channels_first(self.data_format):
            w = tf.reshape(w, shape=(-1, self.num_branches, self.out_channels))
        else:
            w = tf.reshape(w, shape=(-1, self.out_channels, self.num_branches))
        w = self.softmax(w)
        if is_channels_first(self.data_format):
            w = tf.expand_dims(tf.expand_dims(w, 3), 4)
        else:
            w = tf.expand_dims(tf.expand_dims(w, 1), 2)

        y = y * w
        y = tf.math.reduce_sum(y, axis=self.axis)
        return y


class SKNetBottleneck(nn.Layer):
    """
    SKNet bottleneck block for residual path in SKNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck_factor=2,
                 data_format="channels_last",
                 **kwargs):
        super(SKNetBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = SKConvBlock(
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

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SKNetUnit(nn.Layer):
    """
    SKNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 data_format="channels_last",
                 **kwargs):
        super(SKNetUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = SKNetBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
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
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class SKNet(tf.keras.Model):
    """
    SKNet model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

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
        super(SKNet, self).__init__(**kwargs)
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
                stage.add(SKNetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
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


def get_sknet(blocks,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create SKNet model with specific parameters.

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

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported SKNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SKNet(
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


def sknet50(**kwargs):
    """
    SKNet-50 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=50, model_name="sknet50", **kwargs)


def sknet101(**kwargs):
    """
    SKNet-101 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=101, model_name="sknet101", **kwargs)


def sknet152(**kwargs):
    """
    SKNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=152, model_name="sknet152", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        sknet50,
        sknet101,
        sknet152,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sknet50 or weight_count == 27479784)
        assert (model != sknet101 or weight_count == 48736040)
        assert (model != sknet152 or weight_count == 66295656)


if __name__ == "__main__":
    _test()
