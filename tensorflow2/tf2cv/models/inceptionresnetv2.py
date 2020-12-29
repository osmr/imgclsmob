"""
    InceptionResNetV2 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, AvgPool2d, Conv2d, BatchNorm, SimpleSequential, Concurrent, conv1x1, flatten,\
    is_channels_first


class InceptConv(nn.Layer):
    """
    InceptionResNetV2 specific convolution block.

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
            momentum=0.1,
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
    1x1 version of the InceptionResNetV2 specific convolution block.

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
    InceptionResNetV2 specific max pooling branch block.

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
    InceptionResNetV2 specific average pooling branch block.

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
            # count_include_pad=False,
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
    InceptionResNetV2 specific convolutional 1x1 branch block.

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
    InceptionResNetV2 specific convolutional sequence branch block.

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


class InceptionAUnit(nn.Layer):
    """
    InceptionResNetV2 type Inception-A unit.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionAUnit, self).__init__(**kwargs)
        self.scale = 0.17
        in_channels = 320

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=32,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 32),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 48, 64),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            data_format=data_format,
            name="branch3"))
        self.conv = conv1x1(
            in_channels=128,
            out_channels=in_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        identity = x
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class ReductionAUnit(nn.Layer):
    """
    InceptionResNetV2 type Reduction-A unit.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionAUnit, self).__init__(**kwargs)
        in_channels = 320

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
            out_channels_list=(256, 256, 384),
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
    InceptionResNetV2 type Inception-B unit.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionBUnit, self).__init__(**kwargs)
        self.scale = 0.10
        in_channels = 1088

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
            out_channels_list=(128, 160, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            data_format=data_format,
            name="branch2"))
        self.conv = conv1x1(
            in_channels=384,
            out_channels=in_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        identity = x
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class ReductionBUnit(nn.Layer):
    """
    InceptionResNetV2 type Reduction-B unit.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionBUnit, self).__init__(**kwargs)
        in_channels = 1088

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 384),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 288),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 288, 320),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(MaxPoolBranch(
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptionCUnit(nn.Layer):
    """
    InceptionResNetV2 type Inception-C unit.

    Parameters:
    ----------
    scale : float, default 1.0
        Scale value for residual branch.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 scale=0.2,
                 activate=True,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionCUnit, self).__init__(**kwargs)
        self.activate = activate
        self.scale = scale
        in_channels = 2080

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
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 3), (3, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 1), (1, 0)),
            data_format=data_format,
            name="branch2"))
        self.conv = conv1x1(
            in_channels=448,
            out_channels=in_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        if self.activate:
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        identity = x
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        x = self.scale * x + identity
        if self.activate:
            x = self.activ(x)
        return x


class InceptBlock5b(nn.Layer):
    """
    InceptionResNetV2 type Mixed-5b block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(InceptBlock5b, self).__init__(**kwargs)
        in_channels = 192

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=96,
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
            out_channels=64,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptInitBlock(nn.Layer):
    """
    InceptionResNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 data_format="channels_last",
                 **kwargs):
        super(InceptInitBlock, self).__init__(**kwargs)
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
        self.block = InceptBlock5b(
            data_format=data_format,
            name="block")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool1(x)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.pool2(x)
        x = self.block(x, training=training)
        return x


class InceptionResNetV2(tf.keras.Model):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
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
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionResNetV2, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        layers = [10, 21, 11]
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = SimpleSequential(name="features")
        self.features.add(InceptInitBlock(
            in_channels=in_channels,
            data_format=data_format,
            name="init_block"))

        for i, layers_per_stage in enumerate(layers):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j in range(layers_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]
                if (i == len(layers) - 1) and (j == layers_per_stage - 1):
                    stage.add(unit(
                        scale=1.0,
                        activate=False,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                else:
                    stage.add(unit(
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
            self.features.add(stage)
        self.features.add(incept_conv1x1(
            in_channels=2080,
            out_channels=1536,
            data_format=data_format,
            name="final_block"))
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


def get_inceptionresnetv2(model_name=None,
                          pretrained=False,
                          root=os.path.join("~", ".tensorflow", "models"),
                          **kwargs):
    """
    Create InceptionResNetV2 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    net = InceptionResNetV2(**kwargs)

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


def inceptionresnetv2(**kwargs):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_inceptionresnetv2(model_name="inceptionresnetv2", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        inceptionresnetv2,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 299, 299) if is_channels_first(data_format) else (batch, 299, 299, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionresnetv2 or weight_count == 55843464)


if __name__ == "__main__":
    _test()
