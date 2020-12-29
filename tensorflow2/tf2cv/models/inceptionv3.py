"""
    InceptionV3 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.
"""

__all__ = ['InceptionV3', 'inceptionv3']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, AvgPool2d, Conv2d, BatchNorm, SimpleSequential, Concurrent, flatten, is_channels_first,\
    get_channel_axis


class InceptConv(nn.Layer):
    """
    InceptionV3 specific convolution block.

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
        super(InceptConv, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = BatchNorm(
            epsilon=1e-3,
            data_format=data_format,
            name="bn")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activ(x)
        return x


def incept_conv1x1(in_channels,
                   out_channels,
                   data_format="channels_last",
                   **kwargs):
    """
    1x1 version of the InceptionV3 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        data_format=data_format,
        **kwargs)


class MaxPoolBranch(nn.Layer):
    """
    InceptionV3 specific max pooling branch block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(MaxPoolBranch, self).__init__(**kwargs)
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.pool(x)
        return x


class AvgPoolBranch(nn.Layer):
    """
    InceptionV3 specific average pooling branch block.

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
        super(AvgPoolBranch, self).__init__(**kwargs)
        self.pool = AvgPool2d(
            pool_size=3,
            strides=1,
            padding=1,
            data_format=data_format,
            name="pool")
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.pool(x)
        x = self.conv(x, training=training)
        return x


class Conv1x1Branch(nn.Layer):
    """
    InceptionV3 specific convolutional 1x1 branch block.

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
        super(Conv1x1Branch, self).__init__(**kwargs)
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        return x


class ConvSeqBranch(nn.Layer):
    """
    InceptionV3 specific convolutional sequence branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 data_format="channels_last",
                 **kwargs):
        super(ConvSeqBranch, self).__init__(**kwargs)
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = SimpleSequential(name="conv_list")
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.children.append(InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                name="conv{}".format(i + 1)))
            in_channels = out_channels

    def call(self, x, training=None):
        x = self.conv_list(x, training=training)
        return x


class ConvSeq3x3Branch(nn.Layer):
    """
    InceptionV3 specific convolutional sequence branch block with splitting by 3x3.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 data_format="channels_last",
                 **kwargs):
        super(ConvSeq3x3Branch, self).__init__(**kwargs)
        self.data_format = data_format

        self.conv_list = SimpleSequential(name="conv_list")
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.children.append(InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                name="conv{}".format(i + 1)))
            in_channels = out_channels
        self.conv1x3 = InceptConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 3),
            strides=1,
            padding=(0, 1),
            data_format=data_format,
            name="conv1x3")
        self.conv3x1 = InceptConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 1),
            strides=1,
            padding=(1, 0),
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
    InceptionV3 type Inception-A unit.

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
        super(InceptionAUnit, self).__init__(**kwargs)
        assert (out_channels > 224)
        pool_out_channels = out_channels - 224

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=64,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(48, 64),
            kernel_size_list=(1, 5),
            strides_list=(1, 1),
            padding_list=(0, 2),
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(AvgPoolBranch(
            in_channels=in_channels,
            out_channels=pool_out_channels,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class ReductionAUnit(nn.Layer):
    """
    InceptionV3 type Reduction-A unit.

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
        super(ReductionAUnit, self).__init__(**kwargs)
        assert (in_channels == 288)
        assert (out_channels == 768)

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,),
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
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
    InceptionV3 type Inception-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of output channels in the 7x7 branches.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionBUnit, self).__init__(**kwargs)
        assert (in_channels == 768)
        assert (out_channels == 768)

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=192,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, mid_channels, mid_channels, 192),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3)),
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(AvgPoolBranch(
            in_channels=in_channels,
            out_channels=192,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class ReductionBUnit(nn.Layer):
    """
    InceptionV3 type Reduction-B unit.

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
        super(ReductionBUnit, self).__init__(**kwargs)
        assert (in_channels == 768)
        assert (out_channels == 1280)

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 320),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192, 192, 192),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 2),
            padding_list=(0, (0, 3), (3, 0), 0),
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
    InceptionV3 type Inception-C unit.

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
        super(InceptionCUnit, self).__init__(**kwargs)
        assert (out_channels == 2048)

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=320,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,),
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels_list=(448, 384),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(AvgPoolBranch(
            in_channels=in_channels,
            out_channels=192,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptInitBlock(nn.Layer):
    """
    InceptionV3 specific initial block.

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
        super(InceptInitBlock, self).__init__(**kwargs)
        assert (out_channels == 192)

        self.conv1 = InceptConv(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="conv1")
        self.conv2 = InceptConv(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            strides=1,
            padding=0,
            data_format=data_format,
            name="conv2")
        self.conv3 = InceptConv(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            strides=1,
            padding=1,
            data_format=data_format,
            name="conv3")
        self.pool1 = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool1")
        self.conv4 = InceptConv(
            in_channels=64,
            out_channels=80,
            kernel_size=1,
            strides=1,
            padding=0,
            data_format=data_format,
            name="conv4")
        self.conv5 = InceptConv(
            in_channels=80,
            out_channels=192,
            kernel_size=3,
            strides=1,
            padding=0,
            data_format=data_format,
            name="conv5")
        self.pool2 = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool1(x)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.pool2(x)
        return x


class InceptionV3(tf.keras.Model):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    b_mid_channels : list of int
        Number of middle channels for each Inception-B unit.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
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
                 channels,
                 init_block_channels,
                 b_mid_channels,
                 dropout_rate=0.5,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionV3, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = SimpleSequential(name="features")
        self.features.add(InceptInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]
                if unit == InceptionBUnit:
                    stage.add(unit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mid_channels=b_mid_channels[j - 1],
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                else:
                    stage.add(unit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=8,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(nn.Dropout(
            rate=dropout_rate,
            name="dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_inceptionv3(model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".tensorflow", "models"),
                    **kwargs):
    """
    Create InceptionV3 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 192
    channels = [[256, 288, 288],
                [768, 768, 768, 768, 768],
                [1280, 2048, 2048]]
    b_mid_channels = [128, 160, 160, 192]

    net = InceptionV3(
        channels=channels,
        init_block_channels=init_block_channels,
        b_mid_channels=b_mid_channels,
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


def inceptionv3(**kwargs):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv3(model_name="inceptionv3", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        inceptionv3,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 299, 299) if is_channels_first(data_format) else (batch, 299, 299, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionv3 or weight_count == 23834568)


if __name__ == "__main__":
    _test()
