"""
    LEDNet for image segmentation, implemented in TensorFlow.
    Original paper: 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.
"""

__all__ = ['LEDNet', 'lednet_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3, conv1x1_block, conv3x3_block, conv5x5_block, conv7x7_block, ConvBlock, NormActivation,\
    ChannelShuffle, InterpolationBlock, Hourglass, BreakBlock, SimpleSequential, MaxPool2d, is_channels_first,\
    get_channel_axis, get_im_size


class AsymConvBlock(nn.Layer):
    """
    Asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    lw_use_bn : bool, default True
        Whether to use BatchNorm layer (leftwise convolution block).
    rw_use_bn : bool, default True
        Whether to use BatchNorm layer (rightwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    lw_activation : function or str or None, default 'relu'
        Activation function after the leftwise convolution block.
    rw_activation : function or str or None, default 'relu'
        Activation function after the rightwise convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 lw_use_bn=True,
                 rw_use_bn=True,
                 bn_eps=1e-5,
                 lw_activation="relu",
                 rw_activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(AsymConvBlock, self).__init__(**kwargs)
        self.lw_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            strides=1,
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=groups,
            use_bias=use_bias,
            use_bn=lw_use_bn,
            bn_eps=bn_eps,
            activation=lw_activation,
            data_format=data_format,
            name="lw_conv")
        self.rw_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            strides=1,
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            use_bias=use_bias,
            use_bn=rw_use_bn,
            bn_eps=bn_eps,
            activation=rw_activation,
            data_format=data_format,
            name="rw_conv")

    def call(self, x, training=None):
        x = self.lw_conv(x, training=training)
        x = self.rw_conv(x, training=training)
        return x


def asym_conv3x3_block(padding=1,
                       **kwargs):
    """
    3x3 asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    lw_use_bn : bool, default True
        Whether to use BatchNorm layer (leftwise convolution block).
    rw_use_bn : bool, default True
        Whether to use BatchNorm layer (rightwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    lw_activation : function or str or None, default 'relu'
        Activation function after the leftwise convolution block.
    rw_activation : function or str or None, default 'relu'
        Activation function after the rightwise convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return AsymConvBlock(
        kernel_size=3,
        padding=padding,
        **kwargs)


class LEDDownBlock(nn.Layer):
    """
    LEDNet specific downscale block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 correct_size_mismatch,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(LEDDownBlock, self).__init__(**kwargs)
        self.correct_size_mismatch = correct_size_mismatch
        self.data_format = data_format
        self.axis = get_channel_axis(data_format)

        self.pool = MaxPool2d(
            pool_size=2,
            strides=2,
            data_format=data_format,
            name="pool")
        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=(out_channels - in_channels),
            strides=2,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            data_format=data_format,
            name="norm_activ")

    def call(self, x, training=None):
        y1 = self.pool(x)
        y2 = self.conv(x)

        if self.correct_size_mismatch:
            if self.data_format == "channels_last":
                diff_h = y2.size()[1] - y1.size()[1]
                diff_w = y2.size()[2] - y1.size()[2]
            else:
                diff_h = y2.size()[2] - y1.size()[2]
                diff_w = y2.size()[3] - y1.size()[3]
            y1 = nn.ZeroPadding2D(
                padding=((diff_w // 2, diff_w - diff_w // 2), (diff_h // 2, diff_h - diff_h // 2)),
                data_format=self.data_format)(y1)

        x = tf.concat([y2, y1], axis=self.axis)
        x = self.norm_activ(x, training=training)
        return x


class LEDBranch(nn.Layer):
    """
    LEDNet encoder branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(LEDBranch, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = asym_conv3x3_block(
            channels=channels,
            use_bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv1")
        self.conv2 = asym_conv3x3_block(
            channels=channels,
            padding=dilation,
            dilation=dilation,
            use_bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None,
            data_format=data_format,
            name="conv2")
        if self.use_dropout:
            self.dropout = nn.Dropout(
                rate=dropout_rate,
                name="dropout")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        if self.use_dropout:
            x = self.dropout(x, training=training)
        return x


class LEDUnit(nn.Layer):
    """
    LEDNet encoder unit (Split-Shuffle-non-bottleneck).

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(LEDUnit, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        mid_channels = channels // 2

        self.left_branch = LEDBranch(
            channels=mid_channels,
            dilation=dilation,
            dropout_rate=dropout_rate,
            bn_eps=bn_eps,
            data_format=data_format,
            name="left_branch")
        self.right_branch = LEDBranch(
            channels=mid_channels,
            dilation=dilation,
            dropout_rate=dropout_rate,
            bn_eps=bn_eps,
            data_format=data_format,
            name="right_branch")
        self.activ = nn.ReLU()
        self.shuffle = ChannelShuffle(
            channels=channels,
            groups=2,
            data_format=data_format,
            name="shuffle")

    def call(self, x, training=None):
        identity = x

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
        x1 = self.left_branch(x1, training=training)
        x2 = self.right_branch(x2, training=training)
        x = tf.concat([x1, x2], axis=self.axis)

        x = x + identity
        x = self.activ(x)
        x = self.shuffle(x)
        return x


class PoolingBranch(nn.Layer):
    """
    Pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bias : bool
        Whether the layer uses a bias vector.
    bn_eps : float
        Small float added to variance in Batch norm.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias,
                 bn_eps,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(PoolingBranch, self).__init__(**kwargs)
        self.in_size = in_size
        self.data_format = data_format

        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv")
        self.up = InterpolationBlock(
            scale_factor=None,
            out_size=in_size,
            data_format=data_format,
            name="up")

    def call(self, x, training=None):
        in_size = self.in_size if self.in_size is not None else get_im_size(x, data_format=self.data_format)
        x = self.pool(x)
        axis = -1 if is_channels_first(self.data_format) else 1
        x = tf.expand_dims(tf.expand_dims(x, axis=axis), axis=axis)
        x = self.conv(x, training=training)
        x = self.up(x, size=in_size)
        return x


class APN(nn.Layer):
    """
    Attention pyramid network block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(APN, self).__init__(**kwargs)
        self.in_size = in_size
        att_out_channels = 1

        self.pool_branch = PoolingBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=True,
            bn_eps=bn_eps,
            in_size=in_size,
            data_format=data_format,
            name="pool_branch")

        self.body = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="body")

        down_seq = SimpleSequential(name="down_seq")
        down_seq.add(conv7x7_block(
            in_channels=in_channels,
            out_channels=att_out_channels,
            strides=2,
            use_bias=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="down1"))
        down_seq.add(conv5x5_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            strides=2,
            use_bias=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="down2"))
        down3_subseq = SimpleSequential(name="down3")
        down3_subseq.add(conv3x3_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            strides=2,
            use_bias=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv1"))
        down3_subseq.add(conv3x3_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            use_bias=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv2"))
        down_seq.add(down3_subseq)

        up_seq = SimpleSequential(name="up_seq")
        up_seq.add(InterpolationBlock(
            scale_factor=2,
            data_format=data_format,
            name="up1"))
        up_seq.add(InterpolationBlock(
            scale_factor=2,
            data_format=data_format,
            name="up2"))
        up_seq.add(InterpolationBlock(
            scale_factor=2,
            data_format=data_format,
            name="up3"))

        skip_seq = SimpleSequential(name="skip_seq")
        skip_seq.add(BreakBlock(name="skip1"))
        skip_seq.add(conv7x7_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            use_bias=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="skip2"))
        skip_seq.add(conv5x5_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            use_bias=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="skip3"))

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            data_format=data_format,
            name="hg")

    def call(self, x, training=None):
        y = self.pool_branch(x, training=training)
        w = self.hg(x, training=training)
        x = self.body(x, training=training)
        x = x * w
        x = x + y
        return x


class LEDNet(tf.keras.Model):
    """
    LEDNet model from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit.
    dilations : list of int
        Dilations for units.
    dropout_rates : list of list of int
        Dropout rates for each unit in encoder.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images in encoder.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 channels,
                 dilations,
                 dropout_rates,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 data_format="channels_last",
                 **kwargs):
        super(LEDNet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size

        self.encoder = SimpleSequential(name="encoder")
        for i, dilations_per_stage in enumerate(dilations):
            out_channels = channels[i]
            dropout_rate = dropout_rates[i]
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    stage.add(LEDDownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        correct_size_mismatch=correct_size_mismatch,
                        bn_eps=bn_eps,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                    in_channels = out_channels
                else:
                    stage.add(LEDUnit(
                        channels=in_channels,
                        dilation=dilation,
                        dropout_rate=dropout_rate,
                        bn_eps=bn_eps,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
            self.encoder.add(stage)
        self.apn = APN(
            in_channels=in_channels,
            out_channels=classes,
            bn_eps=bn_eps,
            in_size=(in_size[0] // 8, in_size[1] // 8) if fixed_size else None,
            data_format=data_format,
            name="apn")
        self.up = InterpolationBlock(
            scale_factor=8,
            data_format=data_format,
            name="up")

    def call(self, x, training=None):
        x = self.encoder(x, training=training)
        x = self.apn(x, training=training)
        x = self.up(x, training=training)
        return x


def get_lednet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
               **kwargs):
    """
    Create LEDNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    channels = [32, 64, 128]
    dilations = [[0, 1, 1, 1], [0, 1, 1], [0, 1, 2, 5, 9, 2, 5, 9, 17]]
    dropout_rates = [0.03, 0.03, 0.3]
    bn_eps = 1e-3

    net = LEDNet(
        channels=channels,
        dilations=dilations,
        dropout_rates=dropout_rates,
        bn_eps=bn_eps,
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


def lednet_cityscapes(classes=19, **kwargs):
    """
    LEDNet model for Cityscapes from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic
    Segmentation,' https://arxiv.org/abs/1905.02423.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_lednet(classes=classes, model_name="lednet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False
    fixed_size = True
    correct_size_mismatch = False
    in_size = (1024, 2048)
    classes = 19

    models = [
        lednet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size,
                    correct_size_mismatch=correct_size_mismatch, data_format=data_format)

        batch = 4
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes, in_size[0], in_size[1]) if is_channels_first(data_format)
                else tuple(y.shape.as_list()) == (batch, in_size[0], in_size[1], classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lednet_cityscapes or weight_count == 922821)


if __name__ == "__main__":
    _test()
