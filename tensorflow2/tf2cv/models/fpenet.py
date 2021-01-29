"""
    FPENet for image segmentation, implemented in TensorFlow.
    Original paper: 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.
"""

__all__ = ['FPENet', 'fpenet_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, SEBlock, InterpolationBlock, MultiOutputSequential,\
    SimpleSequential, is_channels_first, get_channel_axis


class FPEBlock(nn.Layer):
    """
    FPENet block.

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
        super(FPEBlock, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        dilations = [1, 2, 4, 8]
        assert (channels % len(dilations) == 0)
        mid_channels = channels // len(dilations)

        self.blocks = SimpleSequential(name="blocks")
        for i, dilation in enumerate(dilations):
            self.blocks.add(conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                groups=mid_channels,
                dilation=dilation,
                padding=dilation,
                data_format=data_format,
                name="block{}".format(i + 1)))

    def call(self, x, training=None):
        xs = tf.split(x, num_or_size_splits=len(self.blocks.children), axis=self.axis)
        ys = []
        for bi, xsi in zip(self.blocks.children, xs):
            if len(ys) == 0:
                ys.append(bi(xsi, training=training))
            else:
                ys.append(bi(xsi + ys[-1], training=training))
        x = tf.concat(ys, axis=self.axis)
        return x


class FPEUnit(nn.Layer):
    """
    FPENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int
        Bottleneck factor.
    use_se : bool
        Whether to use SE-module.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck_factor,
                 use_se,
                 data_format="channels_last",
                 **kwargs):
        super(FPEUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)
        self.use_se = use_se
        mid1_channels = in_channels * bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid1_channels,
            strides=strides,
            data_format=data_format,
            name="conv1")
        self.block = FPEBlock(
            channels=mid1_channels,
            data_format=data_format,
            name="blocks")
        self.conv2 = conv1x1_block(
            in_channels=mid1_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv2")
        if self.use_se:
            self.se = SEBlock(
                channels=out_channels,
                data_format=data_format,
                name="se")
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
        x = self.conv1(x, training=training)
        x = self.block(x, training=training)
        x = self.conv2(x, training=training)
        if self.use_se:
            x = self.se(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class FPEStage(nn.Layer):
    """
    FPENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    use_se : bool
        Whether to use SE-module.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 use_se,
                 data_format="channels_last",
                 **kwargs):
        super(FPEStage, self).__init__(**kwargs)
        self.use_block = (layers > 1)

        if self.use_block:
            self.down = FPEUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bottleneck_factor=4,
                use_se=use_se,
                data_format=data_format,
                name="down")
            self.blocks = SimpleSequential(name="blocks")
            for i in range(layers - 1):
                self.blocks.add(FPEUnit(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    bottleneck_factor=1,
                    use_se=use_se,
                    data_format=data_format,
                    name="block{}".format(i + 1)))
        else:
            self.down = FPEUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=1,
                bottleneck_factor=1,
                use_se=use_se,
                data_format=data_format,
                name="down")

    def call(self, x, training=None):
        x = self.down(x, training=training)
        if self.use_block:
            y = self.blocks(x, training=training)
            x = x + y
        return x


class MEUBlock(nn.Layer):
    """
    FPENet specific mutual embedding upsample (MEU) block.

    Parameters:
    ----------
    in_channels_high : int
        Number of input channels for x_high.
    in_channels_low : int
        Number of input channels for x_low.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels_high,
                 in_channels_low,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(MEUBlock, self).__init__(**kwargs)
        self.data_format = data_format
        self.axis = get_channel_axis(data_format)

        self.conv_high = conv1x1_block(
            in_channels=in_channels_high,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv_high")
        self.conv_low = conv1x1_block(
            in_channels=in_channels_low,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv_low")
        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.conv_w_high = conv1x1(
            in_channels=out_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv_w_high")
        self.conv_w_low = conv1x1(
            in_channels=1,
            out_channels=1,
            data_format=data_format,
            name="conv_w_low")
        self.relu = nn.ReLU()
        self.up = InterpolationBlock(
            scale_factor=2,
            data_format=data_format,
            name="up")

    def call(self, x_high, x_low, training=None):
        x_high = self.conv_high(x_high, training=training)
        x_low = self.conv_low(x_low, training=training)

        w_high = self.pool(x_high)
        axis = -1 if is_channels_first(self.data_format) else 1
        w_high = tf.expand_dims(tf.expand_dims(w_high, axis=axis), axis=axis)
        w_high = self.conv_w_high(w_high)
        w_high = self.relu(w_high)
        w_high = tf.nn.sigmoid(w_high)

        w_low = tf.math.reduce_mean(x_low, axis=self.axis, keepdims=True)
        w_low = self.conv_w_low(w_low)
        w_low = tf.nn.sigmoid(w_low)

        x_high = self.up(x_high)

        x_high = x_high * w_low
        x_low = x_low * w_high

        out = x_high + x_low
        return out


class FPENet(tf.keras.Model):
    """
    FPENet model from 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    meu_channels : list of int
        Number of output channels for MEU blocks.
    use_se : bool
        Whether to use SE-module.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 layers,
                 channels,
                 init_block_channels,
                 meu_channels,
                 use_se,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 data_format="channels_last",
                 **kwargs):
        super(FPENet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size
        self.data_format = data_format

        self.stem = conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            strides=2,
            data_format=data_format,
            name="stem")
        in_channels = init_block_channels

        self.encoder = MultiOutputSequential(
            return_last=False,
            name="encoder")
        for i, (layers_i, out_channels) in enumerate(zip(layers, channels)):
            stage = FPEStage(
                in_channels=in_channels,
                out_channels=out_channels,
                layers=layers_i,
                use_se=use_se,
                data_format=data_format,
                name="stage{}".format(i + 1))
            stage.do_output = True
            self.encoder.add(stage)
            in_channels = out_channels

        self.meu1 = MEUBlock(
            in_channels_high=channels[-1],
            in_channels_low=channels[-2],
            out_channels=meu_channels[0],
            data_format=data_format,
            name="meu1")
        self.meu2 = MEUBlock(
            in_channels_high=meu_channels[0],
            in_channels_low=channels[-3],
            out_channels=meu_channels[1],
            data_format=data_format,
            name="meu2")
        in_channels = meu_channels[1]

        self.classifier = conv1x1(
            in_channels=in_channels,
            out_channels=classes,
            use_bias=True,
            data_format=data_format,
            name="classifier")

        self.up = InterpolationBlock(
            scale_factor=2,
            data_format=data_format,
            name="up")

    def call(self, x, training=None):
        x = self.stem(x, training=training)
        y = self.encoder(x, training=training)
        x = self.meu1(y[2], y[1], training=training)
        x = self.meu2(x, y[0], training=training)
        x = self.classifier(x)
        x = self.up(x)
        return x


def get_fpenet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
               **kwargs):
    """
    Create FPENet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    width = 16
    channels = [int(width * (2 ** i)) for i in range(3)]
    init_block_channels = width
    layers = [1, 3, 9]
    meu_channels = [64, 32]
    use_se = False

    net = FPENet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        meu_channels=meu_channels,
        use_se=use_se,
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
                local_model_store_dir_path=root),
            by_name=True,
            skip_mismatch=True)

    return net


def fpenet_cityscapes(classes=19, **kwargs):
    """
    FPENet model for Cityscapes from 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_fpenet(classes=classes, model_name="fpenet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False
    in_size = (1024, 2048)
    classes = 19

    models = [
        fpenet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, data_format=data_format)

        batch = 4
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes, in_size[0], in_size[1]) if is_channels_first(data_format)
                else tuple(y.shape.as_list()) == (batch, in_size[0], in_size[1], classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fpenet_cityscapes or weight_count == 115125)


if __name__ == "__main__":
    _test()
