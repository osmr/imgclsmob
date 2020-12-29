"""
    GhostNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.
"""

__all__ = ['GhostNet', 'ghostnet']

import os
import math
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block,\
    dwsconv3x3_block, SEBlock, SimpleSequential, get_channel_axis, flatten, is_channels_first


class GhostHSigmoid(nn.Layer):
    """
    Approximated sigmoid function, specific for GhostNet.
    """
    def __init__(self, **kwargs):
        super(GhostHSigmoid, self).__init__(**kwargs)

    def call(self, x, training=None):
        return tf.clip_by_value(x, 0.0, 1.0)


class GhostConvBlock(nn.Layer):
    """
    GhostNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(GhostConvBlock, self).__init__(**kwargs)
        self.data_format = data_format
        main_out_channels = math.ceil(0.5 * out_channels)
        cheap_out_channels = out_channels - main_out_channels

        self.main_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=main_out_channels,
            activation=activation,
            data_format=data_format,
            name="main_conv")
        self.cheap_conv = dwconv3x3_block(
            in_channels=main_out_channels,
            out_channels=cheap_out_channels,
            activation=activation,
            data_format=data_format,
            name="cheap_conv")

    def call(self, x, training=None):
        x = self.main_conv(x, training=training)
        y = self.cheap_conv(x, training=training)
        return tf.concat([x, y], axis=get_channel_axis(self.data_format))


class GhostExpBlock(nn.Layer):
    """
    GhostNet expansion block for residual path in GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_kernel3,
                 exp_factor,
                 use_se,
                 data_format="channels_last",
                 **kwargs):
        super(GhostExpBlock, self).__init__(**kwargs)
        self.use_dw_conv = (strides != 1)
        self.use_se = use_se
        mid_channels = int(math.ceil(exp_factor * in_channels))

        self.exp_conv = GhostConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            name="exp_conv")
        if self.use_dw_conv:
            dw_conv_class = dwconv3x3_block if use_kernel3 else dwconv5x5_block
            self.dw_conv = dw_conv_class(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                activation=None,
                data_format=data_format,
                name="dw_conv")
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=4,
                out_activation=GhostHSigmoid(),
                data_format=data_format,
                name="se")
        self.pw_conv = GhostConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="pw_conv")

    def call(self, x, training=None):
        x = self.exp_conv(x, training=training)
        if self.use_dw_conv:
            x = self.dw_conv(x, training=training)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x, training=training)
        return x


class GhostUnit(nn.Layer):
    """
    GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_kernel3,
                 exp_factor,
                 use_se,
                 data_format="channels_last",
                 **kwargs):
        super(GhostUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = GhostExpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            use_kernel3=use_kernel3,
            exp_factor=exp_factor,
            use_se=use_se,
            data_format=data_format,
            name="body")
        if self.resize_identity:
            self.identity_conv = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                pw_activation=None,
                data_format=data_format,
                name="identity_conv")

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        return x


class GhostClassifier(nn.Layer):
    """
    GhostNet classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 data_format="channels_last",
                 **kwargs):
        super(GhostClassifier, self).__init__(**kwargs)
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x)
        return x


class GhostNet(tf.keras.Model):
    """
    GhostNet model from 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
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
                 classifier_mid_channels,
                 kernels3,
                 exp_factors,
                 use_se,
                 first_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(GhostNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            strides=2,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                use_kernel3 = kernels3[i][j] == 1
                exp_factor = exp_factors[i][j]
                use_se_flag = use_se[i][j] == 1
                stage.add(GhostUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor,
                    use_se=use_se_flag,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            data_format=data_format,
            name="final_block"))
        in_channels = final_block_channels
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = GhostClassifier(
            in_channels=in_channels,
            out_channels=classes,
            mid_channels=classifier_mid_channels,
            data_format=data_format,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x, training=training)
        x = flatten(x, self.data_format)
        return x


def get_ghostnet(width_scale=1.0,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create GhostNet model with specific parameters.

    Parameters:
    ----------
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 16
    channels = [[16], [24, 24], [40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160, 160, 160]]
    kernels3 = [[1], [1, 1], [0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    exp_factors = [[1], [3, 3], [3, 3], [6, 2.5, 2.3, 2.3, 6, 6], [6, 6, 6, 6, 6]]
    use_se = [[0], [0, 0], [1, 1], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1]]
    final_block_channels = 960
    classifier_mid_channels = 1280
    first_stride = False

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale, divisor=4) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale, divisor=4)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale, divisor=4)

    net = GhostNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        use_se=use_se,
        first_stride=first_stride,
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


def ghostnet(**kwargs):
    """
    GhostNet model from 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ghostnet(model_name="ghostnet", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        ghostnet,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ghostnet or weight_count == 5180840)


if __name__ == "__main__":
    _test()
