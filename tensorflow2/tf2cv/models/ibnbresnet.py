"""
    IBN(b)-ResNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.
"""

__all__ = ['IBNbResNet', 'ibnb_resnet50', 'ibnb_resnet101', 'ibnb_resnet152']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, InstanceNorm, Conv2d, conv1x1_block, SimpleSequential, flatten, is_channels_first
from .resnet import ResBottleneck


class IBNbConvBlock(nn.Layer):
    """
    IBN(b)-ResNet specific convolution block with Instance normalization and ReLU activation.

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
                 activate=True,
                 data_format="channels_last",
                 **kwargs):
        super(IBNbConvBlock, self).__init__(**kwargs)
        self.activate = activate

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
        self.inst_norm = InstanceNorm(
            scale=True,
            data_format=data_format,
            name="inst_norm")
        if self.activate:
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.inst_norm(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


def ibnb_conv7x7_block(in_channels,
                       out_channels,
                       strides=1,
                       padding=3,
                       use_bias=False,
                       activate=True,
                       data_format="channels_last",
                       **kwargs):
    """
    7x7 version of the IBN(b)-ResNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return IBNbConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        activate=activate,
        data_format=data_format,
        **kwargs)


class IBNbResUnit(nn.Layer):
    """
    IBN(b)-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_inst_norm : bool
        Whether to use instance normalization.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_inst_norm,
                 data_format="channels_last",
                 **kwargs):
        super(IBNbResUnit, self).__init__(**kwargs)
        self.use_inst_norm = use_inst_norm
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = ResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            conv1_stride=False,
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
        if self.use_inst_norm:
            self.inst_norm = InstanceNorm(
                scale=True,
                data_format=data_format,
                name="inst_norm")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        if self.use_inst_norm:
            x = self.inst_norm(x, training=training)
        x = self.activ(x)
        return x


class IBNbResInitBlock(nn.Layer):
    """
    IBN(b)-ResNet specific initial block.

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
        super(IBNbResInitBlock, self).__init__(**kwargs)
        self.conv = ibnb_conv7x7_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            data_format=data_format,
            name="conv")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x


class IBNbResNet(tf.keras.Model):
    """
    IBN(b)-ResNet model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
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
        super(IBNbResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(IBNbResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                use_inst_norm = (i < 2) and (j == len(channels_per_stage) - 1)
                stage.add(IBNbResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    use_inst_norm=use_inst_norm,
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


def get_ibnbresnet(blocks,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".tensorflow", "models"),
                   **kwargs):
    """
    Create IBN(b)-ResNet model with specific parameters.

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
        raise ValueError("Unsupported IBN(b)-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = IBNbResNet(
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


def ibnb_resnet50(**kwargs):
    """
    IBN(b)-ResNet-50 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibnbresnet(blocks=50, model_name="ibnb_resnet50", **kwargs)


def ibnb_resnet101(**kwargs):
    """
    IBN(b)-ResNet-101 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibnbresnet(blocks=101, model_name="ibnb_resnet101", **kwargs)


def ibnb_resnet152(**kwargs):
    """
    IBN(b)-ResNet-152 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibnbresnet(blocks=152, model_name="ibnb_resnet152", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        ibnb_resnet50,
        ibnb_resnet101,
        ibnb_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibnb_resnet50 or weight_count == 25558568)
        assert (model != ibnb_resnet101 or weight_count == 44550696)
        assert (model != ibnb_resnet152 or weight_count == 60194344)


if __name__ == "__main__":
    _test()
