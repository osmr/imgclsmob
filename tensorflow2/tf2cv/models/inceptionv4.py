"""
    InceptionV4 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionV4', 'inceptionv4']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import ConvBlock, conv3x3_block, SimpleSequential, Concurrent, flatten, is_channels_first, get_channel_axis
from .inceptionv3 import MaxPoolBranch, AvgPoolBranch, Conv1x1Branch, ConvSeqBranch


class Conv3x3Branch(nn.Layer):
    """
    InceptionV4 specific convolutional 3x3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(Conv3x3Branch, self).__init__(**kwargs)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        return x


class ConvSeq3x3Branch(nn.Layer):
    """
    InceptionV4 specific convolutional sequence branch block with splitting by 3x3.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels_list : list of tuple of int
        List of numbers of output channels for middle layers.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(ConvSeq3x3Branch, self).__init__(**kwargs)
        self.data_format = data_format

        self.conv_list = SimpleSequential(name="conv_list")
        for i, (mid_channels, kernel_size, strides, padding) in enumerate(zip(
                mid_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.children.append(ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                bn_eps=bn_eps,
                data_format=data_format,
                name="conv{}".format(i + 1)))
            in_channels = mid_channels
        self.conv1x3 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            strides=1,
            padding=(0, 1),
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv1x3")
        self.conv3x1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 1),
            strides=1,
            padding=(1, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv3x1")

    def call(self, x, training=None):
        x = self.conv_list(x, training=training)
        y1 = self.conv1x3(x, training=training)
        y2 = self.conv3x1(x, training=training)
        x = tf.concat([y1, y2], axis=get_channel_axis(self.data_format))
        return x


class InceptionAUnit(nn.Layer):
    """
    InceptionV4 type Inception-A unit.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionAUnit, self).__init__(**kwargs)
        in_channels = 384

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=96,
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(AvgPoolBranch(
            in_channels=in_channels,
            out_channels=96,
            bn_eps=bn_eps,
            count_include_pad=False,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class ReductionAUnit(nn.Layer):
    """
    InceptionV4 type Reduction-A unit.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionAUnit, self).__init__(**kwargs)
        in_channels = 384

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(MaxPoolBranch(
            data_format=data_format,
            name="branch3"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptionBUnit(nn.Layer):
    """
    InceptionV4 type Inception-B unit.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionBUnit, self).__init__(**kwargs)
        in_channels = 1024

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=384,
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192, 224, 224, 256),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3)),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(AvgPoolBranch(
            in_channels=in_channels,
            out_channels=128,
            bn_eps=bn_eps,
            count_include_pad=False,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class ReductionBUnit(nn.Layer):
    """
    InceptionV4 type Reduction-B unit.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionBUnit, self).__init__(**kwargs)
        in_channels = 1024

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 320, 320),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 2),
            padding_list=(0, (0, 3), (3, 0), 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(MaxPoolBranch(
            data_format=data_format,
            name="branch3"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptionCUnit(nn.Layer):
    """
    InceptionV4 type Inception-C unit.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionCUnit, self).__init__(**kwargs)
        in_channels = 1536

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=256,
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384, 448, 512),
            kernel_size_list=(1, (3, 1), (1, 3)),
            strides_list=(1, 1, 1),
            padding_list=(0, (1, 0), (0, 1)),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(AvgPoolBranch(
            in_channels=in_channels,
            out_channels=256,
            bn_eps=bn_eps,
            count_include_pad=False,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptBlock3a(nn.Layer):
    """
    InceptionV4 type Mixed-3a block.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptBlock3a, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(MaxPoolBranch(
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(Conv3x3Branch(
            in_channels=64,
            out_channels=96,
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptBlock4a(nn.Layer):
    """
    InceptionV4 type Mixed-4a block.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptBlock4a, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 64, 64, 96),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 1),
            padding_list=(0, (0, 3), (3, 0), 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptBlock5a(nn.Layer):
    """
    InceptionV4 type Mixed-5a block.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptBlock5a, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv3x3Branch(
            in_channels=192,
            out_channels=192,
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(MaxPoolBranch(
            data_format=data_format,
            name="branch2"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptInitBlock(nn.Layer):
    """
    InceptionV4 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            strides=2,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            strides=1,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=32,
            out_channels=64,
            strides=1,
            padding=1,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv3")
        self.block1 = InceptBlock3a(
            bn_eps=bn_eps,
            data_format=data_format,
            name="block1")
        self.block2 = InceptBlock4a(
            bn_eps=bn_eps,
            data_format=data_format,
            name="block2")
        self.block3 = InceptBlock5a(
            bn_eps=bn_eps,
            data_format=data_format,
            name="block3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        return x


class InceptionV4(tf.keras.Model):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 dropout_rate=0.0,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionV4, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        layers = [4, 8, 4]
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = SimpleSequential(name="features")
        self.features.add(InceptInitBlock(
            in_channels=in_channels,
            bn_eps=bn_eps,
            data_format=data_format,
            name="init_block"))

        for i, layers_per_stage in enumerate(layers):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j in range(layers_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]
                stage.add(unit(
                    bn_eps=bn_eps,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=8,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        if dropout_rate > 0.0:
            self.output1.add(nn.Dropout(
                rate=dropout_rate,
                name="output1/dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=1536,
            name="output1/fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_inceptionv4(model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".tensorflow", "models"),
                    **kwargs):
    """
    Create InceptionV4 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = InceptionV4(**kwargs)

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


def inceptionv4(**kwargs):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv4(model_name="inceptionv4", bn_eps=1e-3, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        inceptionv4,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 299, 299) if is_channels_first(data_format) else (batch, 299, 299, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionv4 or weight_count == 42679816)


if __name__ == "__main__":
    _test()
