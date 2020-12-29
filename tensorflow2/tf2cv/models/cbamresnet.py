"""
    CBAM-ResNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
"""

__all__ = ['CbamResNet', 'cbam_resnet18', 'cbam_resnet34', 'cbam_resnet50', 'cbam_resnet101', 'cbam_resnet152']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv7x7_block, SimpleSequential, flatten, is_channels_first, get_channel_axis
from .resnet import ResInitBlock, ResBlock, ResBottleneck


class MLP(nn.Layer):
    """
    Multilayer perceptron block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 data_format="channels_last",
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Dense(
            units=mid_channels,
            input_dim=channels,
            name="fc1")
        self.activ = nn.ReLU()
        self.fc2 = nn.Dense(
            units=channels,
            input_dim=mid_channels,
            name="fc2")

    def call(self, x, training=None):
        # x = flatten(x, self.data_format)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ChannelGate(nn.Layer):
    """
    CBAM channel gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 data_format="channels_last",
                 **kwargs):
        super(ChannelGate, self).__init__(**kwargs)
        self.data_format = data_format

        self.avg_pool = nn.GlobalAvgPool2D(
            data_format=data_format,
            name="avg_pool")
        self.max_pool = nn.GlobalMaxPool2D(
            data_format=data_format,
            name="max_pool")
        self.mlp = MLP(
            channels=channels,
            reduction_ratio=reduction_ratio,
            data_format=data_format,
            name="mlp")
        self.sigmoid = tf.nn.sigmoid

    def call(self, x, training=None):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        if is_channels_first(self.data_format):
            att = tf.broadcast_to(tf.expand_dims(tf.expand_dims(att, 2), 3), shape=x.shape)
        else:
            att = tf.broadcast_to(tf.expand_dims(tf.expand_dims(att, 1), 2), shape=x.shape)
        x = x * att
        return x


class SpatialGate(nn.Layer):
    """
    CBAM spatial gate block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(SpatialGate, self).__init__(**kwargs)
        self.data_format = data_format

        self.conv = conv7x7_block(
            in_channels=2,
            out_channels=1,
            activation=None,
            data_format=data_format,
            name="conv")
        self.sigmoid = tf.nn.sigmoid

    def call(self, x, training=None):
        axis = get_channel_axis(self.data_format)
        att1 = tf.math.reduce_max(x, axis=axis, keepdims=True)
        att2 = tf.math.reduce_mean(x, axis=axis, keepdims=True)
        att = tf.concat([att1, att2], axis=axis)
        att = self.conv(att, training=training)
        att = tf.broadcast_to(self.sigmoid(att), shape=x.shape)
        x = x * att
        return x


class CbamBlock(nn.Layer):
    """
    CBAM attention block for CBAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 data_format="channels_last",
                 **kwargs):
        super(CbamBlock, self).__init__(**kwargs)
        self.ch_gate = ChannelGate(
            channels=channels,
            reduction_ratio=reduction_ratio,
            data_format=data_format,
            name="ch_gate")
        self.sp_gate = SpatialGate(
            data_format=data_format,
            name="sp_gate")

    def call(self, x, training=None):
        x = self.ch_gate(x, training=training)
        x = self.sp_gate(x, training=training)
        return x


class CbamResUnit(nn.Layer):
    """
    CBAM-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck,
                 data_format="channels_last",
                 **kwargs):
        super(CbamResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                conv1_stride=False,
                data_format=data_format,
                name="body")
        else:
            self.body = ResBlock(
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
        self.cbam = CbamBlock(
            channels=out_channels,
            data_format=data_format,
            name="cbam")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = self.cbam(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class CbamResNet(tf.keras.Model):
    """
    CBAM-ResNet model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
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
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(CbamResNet, self).__init__(**kwargs)
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
                stage.add(CbamResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bottleneck=bottleneck,
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


def get_resnet(blocks,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
               **kwargs):
    """
    Create CBAM-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    use_se : bool
        Whether to use SE block.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported CBAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CbamResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
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


def cbam_resnet18(**kwargs):
    """
    CBAM-ResNet-18 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="cbam_resnet18", **kwargs)


def cbam_resnet34(**kwargs):
    """
    CBAM-ResNet-34 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="cbam_resnet34", **kwargs)


def cbam_resnet50(**kwargs):
    """
    CBAM-ResNet-50 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="cbam_resnet50", **kwargs)


def cbam_resnet101(**kwargs):
    """
    CBAM-ResNet-101 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="cbam_resnet101", **kwargs)


def cbam_resnet152(**kwargs):
    """
    CBAM-ResNet-152 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="cbam_resnet152", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        cbam_resnet18,
        cbam_resnet34,
        cbam_resnet50,
        cbam_resnet101,
        cbam_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cbam_resnet18 or weight_count == 11779392)
        assert (model != cbam_resnet34 or weight_count == 21960468)
        assert (model != cbam_resnet50 or weight_count == 28089624)
        assert (model != cbam_resnet101 or weight_count == 49330172)
        assert (model != cbam_resnet152 or weight_count == 66826848)


if __name__ == "__main__":
    _test()
