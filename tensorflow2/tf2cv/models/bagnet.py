"""
    BagNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.
"""

__all__ = ['BagNet', 'bagnet9', 'bagnet17', 'bagnet33']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, ConvBlock, SimpleSequential, flatten, is_channels_first


class BagNetBottleneck(nn.Layer):
    """
    BagNet bottleneck block for residual path in BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size of the second convolution.
    strides : int or tuple/list of 2 int
        Strides of the second convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 bottleneck_factor=4,
                 data_format="channels_last",
                 **kwargs):
        super(BagNetBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=0,
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


class BagNetUnit(nn.Layer):
    """
    BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size of the second body convolution.
    strides : int or tuple/list of 2 int
        Strides of the second body convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 data_format="channels_last",
                 **kwargs):
        super(BagNetUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = BagNetBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
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
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        if x.shape[-2] != identity.shape[-2]:
            diff = identity.shape[-2] - x.shape[-2]
            if is_channels_first(self.data_format):
                identity = identity[:, :, :-diff, :-diff]
            else:
                identity = identity[:, :-diff, :-diff, :]
        x = x + identity
        x = self.activ(x)
        return x


class BagNetInitBlock(nn.Layer):
    """
    BagNet specific initial block.

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
        super(BagNetInitBlock, self).__init__(**kwargs)
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=0,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class BagNet(tf.keras.Model):
    """
    BagNet model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_pool_size : int
        Size of the pooling windows for final pool.
    normal_kernel_sizes : list of int
        Count of the first units with 3x3 convolution window size for each stage.
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
                 final_pool_size,
                 normal_kernel_sizes,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(BagNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(BagNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != len(channels) - 1) else 1
                kernel_size = 3 if j < normal_kernel_sizes[i] else 1
                stage.add(BagNetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=final_pool_size,
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


def get_bagnet(field,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
               **kwargs):
    """
    Create BagNet model with specific parameters.

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

    layers = [3, 4, 6, 3]

    if field == 9:
        normal_kernel_sizes = [1, 1, 0, 0]
        final_pool_size = 27
    elif field == 17:
        normal_kernel_sizes = [1, 1, 1, 0]
        final_pool_size = 26
    elif field == 33:
        normal_kernel_sizes = [1, 1, 1, 1]
        final_pool_size = 24
    else:
        raise ValueError("Unsupported BagNet with field: {}".format(field))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BagNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_pool_size=final_pool_size,
        normal_kernel_sizes=normal_kernel_sizes,
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


def bagnet9(**kwargs):
    """
    BagNet-9 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=9, model_name="bagnet9", **kwargs)


def bagnet17(**kwargs):
    """
    BagNet-17 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=17, model_name="bagnet17", **kwargs)


def bagnet33(**kwargs):
    """
    BagNet-33 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=33, model_name="bagnet33", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        bagnet9,
        bagnet17,
        bagnet33,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bagnet9 or weight_count == 15688744)
        assert (model != bagnet17 or weight_count == 16213032)
        assert (model != bagnet33 or weight_count == 18310184)


if __name__ == "__main__":
    _test()
