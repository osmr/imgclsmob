"""
    BAM-ResNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.
"""

__all__ = ['BamResNet', 'bam_resnet18', 'bam_resnet34', 'bam_resnet50', 'bam_resnet101', 'bam_resnet152']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, BatchNorm, SimpleSequential, flatten,\
    is_channels_first
from .resnet import ResInitBlock, ResUnit


class DenseBlock(nn.Layer):
    """
    Standard dense block with Batch normalization and ReLU activation.

    Parameters:
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.fc = nn.Dense(
            units=out_channels,
            input_dim=in_channels,
            name="fc")
        self.bn = BatchNorm(
            data_format=data_format,
            name="bn")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        x = self.activ(x)
        return x


class ChannelGate(nn.Layer):
    """
    BAM channel gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_layers : int, default 1
        Number of dense blocks.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 num_layers=1,
                 data_format="channels_last",
                 **kwargs):
        super(ChannelGate, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = channels // reduction_ratio

        self.pool = nn.GlobalAvgPool2D(
            data_format=data_format,
            name="pool")
        self.flatten = nn.Flatten()
        self.init_fc = DenseBlock(
            in_channels=channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="init_fc")
        self.main_fcs = SimpleSequential(name="main_fcs")
        for i in range(num_layers - 1):
            self.main_fcs.children.append(DenseBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                data_format=data_format,
                name="fc{}".format(i + 1)))
        self.final_fc = nn.Dense(
            units=channels,
            input_dim=mid_channels,
            name="final_fc")

    def call(self, x, training=None):
        input = x
        x = self.pool(x)
        x = self.flatten(x)
        x = self.init_fc(x)
        x = self.main_fcs(x, training=training)
        x = self.final_fc(x)
        if is_channels_first(self.data_format):
            x = tf.broadcast_to(tf.expand_dims(tf.expand_dims(x, 2), 3), shape=input.shape)
        else:
            x = tf.broadcast_to(tf.expand_dims(tf.expand_dims(x, 1), 2), shape=input.shape)
        return x


class SpatialGate(nn.Layer):
    """
    BAM spatial gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_dil_convs : int, default 2
        Number of dilated convolutions.
    dilation : int, default 4
        Dilation/padding value for corresponding convolutions.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 num_dil_convs=2,
                 dilation=4,
                 data_format="channels_last",
                 **kwargs):
        super(SpatialGate, self).__init__(**kwargs)
        mid_channels = channels // reduction_ratio

        self.init_conv = conv1x1_block(
            in_channels=channels,
            out_channels=mid_channels,
            strides=1,
            use_bias=True,
            data_format=data_format,
            name="init_conv")
        self.dil_convs = SimpleSequential(name="dil_convs")
        for i in range(num_dil_convs):
            self.dil_convs.children.append(conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=1,
                padding=dilation,
                dilation=dilation,
                use_bias=True,
                data_format=data_format,
                name="conv{}".format(i + 1)))
        self.final_conv = conv1x1(
            in_channels=mid_channels,
            out_channels=1,
            strides=1,
            use_bias=True,
            data_format=data_format,
            name="final_conv")

    def call(self, x, training=None):
        input = x
        x = self.init_conv(x, training=training)
        x = self.dil_convs(x, training=training)
        x = self.final_conv(x)
        x = tf.broadcast_to(x, shape=input.shape)
        return x


class BamBlock(nn.Layer):
    """
    BAM attention block for BAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 data_format="channels_last",
                 **kwargs):
        super(BamBlock, self).__init__(**kwargs)
        self.ch_att = ChannelGate(
            channels=channels,
            data_format=data_format,
            name="ch_att")
        self.sp_att = SpatialGate(
            channels=channels,
            data_format=data_format,
            name="sp_att")
        self.sigmoid = tf.nn.sigmoid

    def call(self, x, training=None):
        att = 1 + self.sigmoid(self.ch_att(x, training=training) * self.sp_att(x, training=training))
        x = x * att
        return x


class BamResUnit(nn.Layer):
    """
    BAM-ResNet unit.

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
        super(BamResUnit, self).__init__(**kwargs)
        self.use_bam = (strides != 1)

        if self.use_bam:
            self.bam = BamBlock(
                channels=in_channels,
                data_format=data_format,
                name="bam")
        self.res_unit = ResUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            bottleneck=bottleneck,
            conv1_stride=False,
            data_format=data_format,
            name="res_unit")

    def call(self, x, training=None):
        if self.use_bam:
            x = self.bam(x, training=training)
        x = self.res_unit(x, training=training)
        return x


class BamResNet(tf.keras.Model):
    """
    BAM-ResNet model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

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
        super(BamResNet, self).__init__(**kwargs)
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
                stage.add(BamResUnit(
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
    Create BAM-ResNet model with specific parameters.

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
        raise ValueError("Unsupported BAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BamResNet(
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


def bam_resnet18(**kwargs):
    """
    BAM-ResNet-18 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="bam_resnet18", **kwargs)


def bam_resnet34(**kwargs):
    """
    BAM-ResNet-34 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="bam_resnet34", **kwargs)


def bam_resnet50(**kwargs):
    """
    BAM-ResNet-50 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="bam_resnet50", **kwargs)


def bam_resnet101(**kwargs):
    """
    BAM-ResNet-101 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="bam_resnet101", **kwargs)


def bam_resnet152(**kwargs):
    """
    BAM-ResNet-152 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="bam_resnet152", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        bam_resnet18,
        bam_resnet34,
        bam_resnet50,
        bam_resnet101,
        bam_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bam_resnet18 or weight_count == 11712503)
        assert (model != bam_resnet34 or weight_count == 21820663)
        assert (model != bam_resnet50 or weight_count == 25915099)
        assert (model != bam_resnet101 or weight_count == 44907227)
        assert (model != bam_resnet152 or weight_count == 60550875)


if __name__ == "__main__":
    _test()
