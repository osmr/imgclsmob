"""
    IGCV3 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.
"""

__all__ = ['IGCV3', 'igcv3_w1', 'igcv3_w3d4', 'igcv3_wd2', 'igcv3_wd4']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, ChannelShuffle, ReLU6, SimpleSequential, flatten


class InvResUnit(nn.Layer):
    """
    So-called 'Inverted Residual Unit' layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    expansion : bool
        Whether do expansion of channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 expansion,
                 data_format="channels_last",
                 **kwargs):
        super(InvResUnit, self).__init__(**kwargs)
        self.residual = (in_channels == out_channels) and (strides == 1)
        mid_channels = in_channels * 6 if expansion else in_channels
        groups = 2

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=groups,
            activation=None,
            data_format=data_format,
            name="conv1")
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=groups,
            data_format=data_format,
            name="c_shuffle")
        self.conv2 = dwconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            activation=ReLU6(),
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            groups=groups,
            activation=None,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        if self.residual:
            identity = x
        x = self.conv1(x, training=training)
        x = self.c_shuffle(x)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        if self.residual:
            x = x + identity
        return x


class IGCV3(tf.keras.Model):
    """
    IGCV3 model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
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
                 final_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(IGCV3, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            strides=2,
            activation=ReLU6(),
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                stage.add(InvResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    expansion=expansion,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            activation=ReLU6(),
            data_format=data_format,
            name="final_block"))
        in_channels = final_block_channels
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


def get_igcv3(width_scale,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create IGCV3-D model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 4, 6, 8, 6, 6, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [[]])

    if width_scale != 1.0:
        def make_even(x):
            return x if (x % 2 == 0) else x + 1
        channels = [[make_even(int(cij * width_scale)) for cij in ci] for ci in channels]
        init_block_channels = make_even(int(init_block_channels * width_scale))
        if width_scale > 1.0:
            final_block_channels = make_even(int(final_block_channels * width_scale))

    net = IGCV3(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
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


def igcv3_w1(**kwargs):
    """
    IGCV3-D 1.0x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=1.0, model_name="igcv3_w1", **kwargs)


def igcv3_w3d4(**kwargs):
    """
    IGCV3-D 0.75x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.75, model_name="igcv3_w3d4", **kwargs)


def igcv3_wd2(**kwargs):
    """
    IGCV3-D 0.5x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.5, model_name="igcv3_wd2", **kwargs)


def igcv3_wd4(**kwargs):
    """
    IGCV3-D 0.25x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.25, model_name="igcv3_wd4", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        igcv3_w1,
        igcv3_w3d4,
        igcv3_wd2,
        igcv3_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != igcv3_w1 or weight_count == 3491688)
        assert (model != igcv3_w3d4 or weight_count == 2638084)
        assert (model != igcv3_wd2 or weight_count == 1985528)
        assert (model != igcv3_wd4 or weight_count == 1534020)


if __name__ == "__main__":
    _test()
