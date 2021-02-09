"""
    DABNet for image segmentation, implemented in TensorFlow.
    Original paper: 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.
"""

__all__ = ['DABNet', 'dabnet_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv3x3, conv3x3_block, ConvBlock, NormActivation, Concurrent, InterpolationBlock,\
    DualPathSequential, SimpleSequential, is_channels_first, get_im_size, PReLU2, MaxPool2d, AvgPool2d, get_channel_axis


class DwaConvBlock(nn.Layer):
    """
    Depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(DwaConvBlock, self).__init__(**kwargs)
        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            strides=strides,
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=channels,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=activation,
            data_format=data_format,
            name="conv1")
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            strides=strides,
            padding=(0, padding),
            dilation=(1, dilation),
            groups=channels,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=activation,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


def dwa_conv3x3_block(channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      use_bias=False,
                      use_bn=True,
                      bn_eps=1e-5,
                      activation="relu",
                      data_format="channels_last",
                      **kwargs):
    """
    3x3 version of the depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwaConvBlock(
        channels=channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


class DABBlock(nn.Layer):
    """
    DABNet specific base block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for a dilated branch in the unit.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 dilation,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(DABBlock, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        mid_channels = channels // 2

        self.norm_activ1 = NormActivation(
            in_channels=channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="norm_activ1")
        self.conv1 = conv3x3_block(
            in_channels=channels,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(mid_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv1")

        self.branches = Concurrent(
            stack=True,
            data_format=data_format,
            name="branches")
        self.branches.add(dwa_conv3x3_block(
            channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(mid_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="branches1"))
        self.branches.add(dwa_conv3x3_block(
            channels=mid_channels,
            padding=dilation,
            dilation=dilation,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(mid_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="branches2"))

        self.norm_activ2 = NormActivation(
            in_channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(mid_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="norm_activ2")
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=channels,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        identity = x

        x = self.norm_activ1(x, training=training)
        x = self.conv1(x, training=training)

        x = self.branches(x, training=training)
        x = tf.math.reduce_sum(x, axis=self.axis)

        x = self.norm_activ2(x, training=training)
        x = self.conv2(x)

        x = x + identity
        return x


class DownBlock(nn.Layer):
    """
    DABNet specific downsample block for the main branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(DownBlock, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        self.expand = (in_channels < out_channels)
        mid_channels = out_channels - in_channels if self.expand else out_channels

        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv")
        if self.expand:
            self.pool = MaxPool2d(
                pool_size=2,
                strides=2,
                data_format=data_format,
                name="pool")
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="norm_activ")

    def call(self, x, training=None):
        y = self.conv(x)

        if self.expand:
            z = self.pool(x)
            y = tf.concat([y, z], axis=self.axis)

        y = self.norm_activ(y, training=training)
        return y


class DABUnit(nn.Layer):
    """
    DABNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilations : list of int
        Dilations for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(DABUnit, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        mid_channels = out_channels // 2

        self.down = DownBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            data_format=data_format,
            name="down")
        self.blocks = SimpleSequential(name="blocks")
        for i, dilation in enumerate(dilations):
            self.blocks.add(DABBlock(
                channels=mid_channels,
                dilation=dilation,
                bn_eps=bn_eps,
                data_format=data_format,
                name="block{}".format(i + 1)))

    def call(self, x, training=None):
        x = self.down(x, training=training)
        y = self.blocks(x, training=training)
        x = tf.concat([y, x], axis=self.axis)
        return x


class DABStage(nn.Layer):
    """
    DABNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    dilations : list of int
        Dilations for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 dilations,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(DABStage, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        self.use_unit = (len(dilations) > 0)

        self.x_down = AvgPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="x_down")

        if self.use_unit:
            self.unit = DABUnit(
                in_channels=y_in_channels,
                out_channels=(y_out_channels - x_channels),
                dilations=dilations,
                bn_eps=bn_eps,
                data_format=data_format,
                name="unit")

        self.norm_activ = NormActivation(
            in_channels=y_out_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(y_out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="norm_activ")

    def call(self, y, x, training=None):
        x = self.x_down(x)
        if self.use_unit:
            y = self.unit(y, training=training)
        y = tf.concat([y, x], axis=self.axis)
        y = self.norm_activ(y, training=training)
        return y, x


class DABInitBlock(nn.Layer):
    """
    DABNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(DABInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class DABNet(tf.keras.Model):
    """
    DABNet model from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of list of int
        Dilations for blocks.
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
                 init_block_channels,
                 dilations,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 data_format="channels_last",
                 **kwargs):
        super(DABNet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size
        self.data_format = data_format

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=0,
            name="features")
        self.features.add(DABInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps,
            data_format=data_format,
            name="init_block"))
        y_in_channels = init_block_channels

        for i, (y_out_channels, dilations_i) in enumerate(zip(channels, dilations)):
            self.features.add(DABStage(
                x_channels=in_channels,
                y_in_channels=y_in_channels,
                y_out_channels=y_out_channels,
                dilations=dilations_i,
                bn_eps=bn_eps,
                data_format=data_format,
                name="stage{}".format(i + 1)))
            y_in_channels = y_out_channels

        self.classifier = conv1x1(
            in_channels=y_in_channels,
            out_channels=classes,
            data_format=data_format,
            name="classifier")

        self.up = InterpolationBlock(
            scale_factor=8,
            data_format=data_format,
            name="up")

    def call(self, x, training=None):
        in_size = self.in_size if self.fixed_size else get_im_size(x, data_format=self.data_format)
        y = self.features(x, x, training=training)
        y = self.classifier(y)
        y = self.up(y, size=in_size)
        return y


def get_dabnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
               **kwargs):
    """
    Create DABNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    channels = [35, 131, 259]
    dilations = [[], [2, 2, 2], [4, 4, 8, 8, 16, 16]]
    bn_eps = 1e-3

    net = DABNet(
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
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


def dabnet_cityscapes(classes=19, **kwargs):
    """
    DABNet model for Cityscapes from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dabnet(classes=classes, model_name="dabnet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False
    in_size = (1024, 2048)
    classes = 19

    models = [
        dabnet_cityscapes,
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
        assert (model != dabnet_cityscapes or weight_count == 756643)


if __name__ == "__main__":
    _test()
