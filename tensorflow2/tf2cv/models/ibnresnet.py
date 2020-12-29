"""
    IBN-ResNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.
"""

__all__ = ['IBNResNet', 'ibn_resnet50', 'ibn_resnet101', 'ibn_resnet152']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, BatchNorm, conv1x1_block, conv3x3_block, IBN, SimpleSequential, flatten, is_channels_first
from .resnet import ResInitBlock


class IBNConvBlock(nn.Layer):
    """
    IBN-Net specific convolution block with BN/IBN normalization and ReLU activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_ibn : bool, default False
        Whether use Instance-Batch Normalization.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_ibn=False,
                 activate=True,
                 data_format="channels_last",
                 **kwargs):
        super(IBNConvBlock, self).__init__(**kwargs)
        self.activate = activate
        self.use_ibn = use_ibn

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            data_format=data_format,
            name="conv")
        if self.use_ibn:
            self.ibn = IBN(
                channels=out_channels,
                data_format=data_format,
                name="ibn")
        else:
            self.bn = BatchNorm(
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        if self.use_ibn:
            x = self.ibn(x, training=training)
        else:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


def ibn_conv1x1_block(in_channels,
                      out_channels,
                      strides=1,
                      groups=1,
                      use_bias=False,
                      use_ibn=False,
                      activate=True,
                      data_format="channels_last",
                      **kwargs):
    """
    1x1 version of the IBN-Net specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_ibn : bool, default False
        Whether use Instance-Batch Normalization.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return IBNConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        use_ibn=use_ibn,
        activate=activate,
        data_format=data_format,
        **kwargs)


class IBNResBottleneck(nn.Layer):
    """
    IBN-ResNet bottleneck block for residual path in IBN-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_ibn : bool
        Whether to use IBN normalization in the first convolution layer of the block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 conv1_ibn,
                 data_format="channels_last",
                 **kwargs):
        super(IBNResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        self.conv1 = ibn_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_ibn=conv1_ibn,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
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
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class IBNResUnit(nn.Layer):
    """
    IBN-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_ibn : bool
        Whether to use IBN normalization in the first convolution layer of the block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 conv1_ibn,
                 data_format="channels_last",
                 **kwargs):
        super(IBNResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = IBNResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            conv1_ibn=conv1_ibn,
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
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class IBNResNet(tf.keras.Model):
    """
    IBN-ResNet model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

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
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(IBNResNet, self).__init__(**kwargs)
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
                conv1_ibn = (out_channels < 2048)
                stage.add(IBNResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    conv1_ibn=conv1_ibn,
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


def get_ibnresnet(blocks,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
                  **kwargs):
    """
    Create IBN-ResNet model with specific parameters.

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
        raise ValueError("Unsupported IBN-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = IBNResNet(
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


def ibn_resnet50(**kwargs):
    """
    IBN-ResNet-50 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibnresnet(blocks=50, model_name="ibn_resnet50", **kwargs)


def ibn_resnet101(**kwargs):
    """
    IBN-ResNet-101 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibnresnet(blocks=101, model_name="ibn_resnet101", **kwargs)


def ibn_resnet152(**kwargs):
    """
    IBN-ResNet-152 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibnresnet(blocks=152, model_name="ibn_resnet152", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        ibn_resnet50,
        ibn_resnet101,
        ibn_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibn_resnet50 or weight_count == 25557032)
        assert (model != ibn_resnet101 or weight_count == 44549160)
        assert (model != ibn_resnet152 or weight_count == 60192808)


if __name__ == "__main__":
    _test()
