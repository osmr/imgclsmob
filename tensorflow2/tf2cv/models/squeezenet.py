"""
    SqueezeNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,'
    https://arxiv.org/abs/1602.07360.
"""

__all__ = ['SqueezeNet', 'squeezenet_v1_0', 'squeezenet_v1_1', 'squeezeresnet_v1_0', 'squeezeresnet_v1_1']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import get_channel_axis, Conv2d, MaxPool2d, SimpleSequential, flatten


class FireConv(nn.Layer):
    """
    SqueezeNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 data_format="channels_last",
                 **kwargs):
        super(FireConv, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
            name="conv")
        self.activ = nn.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class FireUnit(nn.Layer):
    """
    SqueezeNet unit, so-called 'Fire' unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    squeeze_channels : int
        Number of output channels for squeeze convolution blocks.
    expand1x1_channels : int
        Number of output channels for expand 1x1 convolution blocks.
    expand3x3_channels : int
        Number of output channels for expand 3x3 convolution blocks.
    residual : bool
        Whether use residual connection.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 squeeze_channels,
                 expand1x1_channels,
                 expand3x3_channels,
                 residual,
                 data_format="channels_last",
                 **kwargs):
        super(FireUnit, self).__init__(**kwargs)
        self.residual = residual
        self.data_format = data_format

        self.squeeze = FireConv(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            padding=0,
            data_format=data_format,
            name="squeeze")
        self.expand1x1 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand1x1_channels,
            kernel_size=1,
            padding=0,
            data_format=data_format,
            name="expand1x1")
        self.expand3x3 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand3x3_channels,
            kernel_size=3,
            padding=1,
            data_format=data_format,
            name="expand3x3")

    def call(self, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = tf.concat([y1, y2], axis=get_channel_axis(self.data_format))
        if self.residual:
            out = out + identity
        return out


class SqueezeInitBlock(nn.Layer):
    """
    SqueezeNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 data_format="channels_last",
                 **kwargs):
        super(SqueezeInitBlock, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=2,
            data_format=data_format,
            name="conv")
        self.activ = nn.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class SqueezeNet(tf.keras.Model):
    """
    SqueezeNet model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,'
    https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    residuals : bool
        Whether to use residual units.
    init_block_kernel_size : int or tuple/list of 2 int
        The dimensions of the convolution window for the initial unit.
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
                 residuals,
                 init_block_kernel_size,
                 init_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(SqueezeInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=init_block_kernel_size,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            stage.add(MaxPool2d(
                pool_size=3,
                strides=2,
                ceil_mode=True,
                data_format=data_format,
                name="pool{}".format(i + 1)))
            for j, out_channels in enumerate(channels_per_stage):
                expand_channels = out_channels // 2
                squeeze_channels = out_channels // 8
                stage.add(FireUnit(
                    in_channels=in_channels,
                    squeeze_channels=squeeze_channels,
                    expand1x1_channels=expand_channels,
                    expand3x3_channels=expand_channels,
                    residual=((residuals is not None) and (residuals[i][j] == 1)),
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.Dropout(
            rate=0.5,
            name="dropout"))

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(Conv2d(
            in_channels=in_channels,
            out_channels=classes,
            kernel_size=1,
            data_format=data_format,
            name="final_conv"))
        self.output1.add(nn.ReLU())
        self.output1.add(nn.AveragePooling2D(
            pool_size=13,
            strides=1,
            data_format=data_format,
            name="final_pool"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        x = flatten(x, self.data_format)
        return x


def get_squeezenet(version,
                   residual=False,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".tensorflow", "models"),
                   **kwargs):
    """
    Create SqueezeNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('1.0' or '1.1').
    residual : bool, default False
        Whether to use residual connections.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if version == "1.0":
        channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
        residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]
        init_block_kernel_size = 7
        init_block_channels = 96
    elif version == "1.1":
        channels = [[128, 128], [256, 256], [384, 384, 512, 512]]
        residuals = [[0, 1], [0, 1], [0, 1, 0, 1]]
        init_block_kernel_size = 3
        init_block_channels = 64
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    if not residual:
        residuals = None

    net = SqueezeNet(
        channels=channels,
        residuals=residuals,
        init_block_kernel_size=init_block_kernel_size,
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


def squeezenet_v1_0(**kwargs):
    """
    SqueezeNet 'vanilla' model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
    size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.0", residual=False, model_name="squeezenet_v1_0", **kwargs)


def squeezenet_v1_1(**kwargs):
    """
    SqueezeNet v1.1 model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
    size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.1", residual=False, model_name="squeezenet_v1_1", **kwargs)


def squeezeresnet_v1_0(**kwargs):
    """
    SqueezeNet model with residual connections from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
    <0.5MB model size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.0", residual=True, model_name="squeezeresnet_v1_0", **kwargs)


def squeezeresnet_v1_1(**kwargs):
    """
    SqueezeNet v1.1 model with residual connections from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.1", residual=True, model_name="squeezeresnet_v1_1", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        squeezenet_v1_0,
        squeezenet_v1_1,
        squeezeresnet_v1_0,
        squeezeresnet_v1_1,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != squeezenet_v1_0 or weight_count == 1248424)
        assert (model != squeezenet_v1_1 or weight_count == 1235496)
        assert (model != squeezeresnet_v1_0 or weight_count == 1248424)
        assert (model != squeezeresnet_v1_1 or weight_count == 1235496)


if __name__ == "__main__":
    _test()
