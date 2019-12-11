"""
    Common routines for models in TensorFlow 2.0.
"""

__all__ = ['is_channels_first', 'get_channel_axis', 'round_channels', 'ReLU6', 'HSwish', 'get_activation_layer',
           'flatten', 'GluonBatchNormalization', 'MaxPool2d', 'AvgPool2d', 'Conv2d', 'conv1x1', 'conv3x3',
           'depthwise_conv3x3', 'ConvBlock', 'conv1x1_block', 'conv3x3_block', 'conv5x5_block', 'conv7x7_block',
           'dwconv3x3_block', 'dwconv5x5_block', 'PreConvBlock', 'pre_conv1x1_block', 'pre_conv3x3_block',
           'ChannelShuffle', 'ChannelShuffle2', 'SEBlock']

import math
from inspect import isfunction
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn


def is_channels_first(data_format):
    """
    Is tested data format channels first.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    bool
        A flag.
    """
    return data_format == "channels_first"


def get_channel_axis(data_format):
    """
    Get channel axis.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    int
        Channel axis.
    """
    return 1 if is_channels_first(data_format) else -1


def round_channels(channels,
                   divisor=8):
    """
    Round weighted channel number (make divisible operation).

    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


class ReLU6(nn.Layer):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x)


class HSigmoid(nn.Layer):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x + 3.0) / 6.0


class HSwish(nn.Layer):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Layer
        Activation function or name of activation function.

    Returns
    -------
    nn.Layer
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "relu6":
            return ReLU6()
        elif activation == "swish":
            return nn.Swish()
        elif activation == "hswish":
            return HSwish()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Layer))
        return activation


def flatten(x,
            data_format):
    """
    Flattens the input to two dimensional.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if not is_channels_first(data_format):
        x = tf.transpose(x, perm=(0, 3, 1, 2))
    x = tf.reshape(x, shape=(-1, np.prod(x.get_shape().as_list()[1:])))
    return x


class GluonBatchNormalization(nn.BatchNormalization):
    """
    MXNet/Gluon-like batch normalization.

    Parameters:
    ----------
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(GluonBatchNormalization, self).__init__(
            axis=get_channel_axis(data_format),
            momentum=momentum,
            epsilon=epsilon,
            **kwargs)

    # def call(self, inputs, training=None):
    #     super(GluonBatchNormalization, self).call(inputs, training)


class MaxPool2d(nn.Layer):
    """
    Max pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 pool_size,
                 strides,
                 padding=0,
                 ceil_mode=False,
                 data_format="channels_last",
                 **kwargs):
        super(MaxPool2d, self).__init__(**kwargs)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.ceil_mode = ceil_mode
        self.use_pad = (padding[0] > 0) or (padding[1] > 0)

        if self.ceil_mode:
            self.padding = padding
            self.pool_size = pool_size
            self.strides = strides
            self.data_format = data_format
        else:
            if is_channels_first(data_format):
                self.paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
            else:
                self.paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]

        self.pool = nn.MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            padding="valid",
            data_format=data_format)

    def call(self, x):
        if self.ceil_mode:
            padding = self.padding
            height = int(x.shape[2])
            out_height = float(height + 2 * padding[0] - self.pool_size[0]) / self.strides[0] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            width = int(x.shape[3])
            out_width = float(width + 2 * padding[1] - self.pool_size[1]) / self.strides[1] + 1.0
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)
            if (padding[0] > 0) or (padding[1] > 0):
                if is_channels_first(self.data_format):
                    paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
                else:
                    paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]
                x = tf.pad(x, paddings=paddings_tf)
        elif self.use_pad:
            if self.use_pad:
                x = tf.pad(x, paddings=self.paddings_tf)

        x = self.pool(x)
        return x


class AvgPool2d(nn.Layer):
    """
    Average pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 pool_size,
                 strides,
                 padding=0,
                 ceil_mode=False,
                 data_format="channels_last",
                 **kwargs):
        super(AvgPool2d, self).__init__(**kwargs)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.ceil_mode = ceil_mode
        self.use_pad = (padding[0] > 0) or (padding[1] > 0)

        if self.ceil_mode:
            self.padding = padding
            self.pool_size = pool_size
            self.strides = strides
            self.data_format = data_format
        else:
            if is_channels_first(data_format):
                self.paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
            else:
                self.paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]

        self.pool = nn.AveragePooling2D(
            pool_size=pool_size,
            strides=1,
            padding="valid",
            data_format=data_format,
            name="pool")

        self.use_stride_pool = (strides[0] > 1) or (strides[1] > 1)
        if self.use_stride_pool:
            self.stride_pool = nn.AveragePooling2D(
                pool_size=1,
                strides=strides,
                padding="valid",
                data_format=data_format,
                name="stride_pool")

    def call(self, x):
        if self.ceil_mode:
            padding = self.padding
            height = int(x.shape[2])
            out_height = float(height + 2 * padding[0] - self.pool_size[0]) / self.strides[0] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            width = int(x.shape[3])
            out_width = float(width + 2 * padding[1] - self.pool_size[1]) / self.strides[1] + 1.0
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)
            if (padding[0] > 0) or (padding[1] > 0):
                if is_channels_first(self.data_format):
                    paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
                else:
                    paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]
                x = tf.pad(x, paddings=paddings_tf)
        elif self.use_pad:
            if self.use_pad:
                x = tf.pad(x, paddings=self.paddings_tf)

        x = self.pool(x)
        if self.use_stride_pool:
            x = self.stride_pool(x)
        return x


class Conv2d(nn.Layer):
    """
    Standard convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 data_format="channels_last",
                 **kwargs):
        super(Conv2d, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.data_format = data_format
        self.use_conv = (groups == 1)
        self.use_dw_conv = (groups > 1) and (groups == out_channels) and (out_channels == in_channels)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.use_pad = (padding[0] > 0) or (padding[1] > 0)
        if self.use_pad:
            self.pad = nn.ZeroPadding2D(
                padding=padding,
                data_format=data_format)

        if self.use_conv:
            self.conv = nn.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                name="conv")
        elif self.use_dw_conv:
            assert (dilation[0] == 1) and (dilation[1] == 1)
            self.dw_conv = nn.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                name="dw_conv")
        else:
            assert (groups > 1)
            assert (in_channels % groups == 0)
            assert (out_channels % groups == 0)
            self.groups = groups
            self.convs = []
            for i in range(groups):
                self.convs.append(nn.Conv2D(
                    filters=(out_channels // groups),
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    data_format=data_format,
                    dilation_rate=dilation,
                    use_bias=use_bias,
                    name="convgroup{}".format(i + 1)))

    def call(self, x):
        if self.use_pad:
            x = self.pad(x)
        if self.use_conv:
            x = self.conv(x)
        elif self.use_dw_conv:
            x = self.dw_conv(x)
        else:
            yy = []
            xx = tf.split(x, num_or_size_splits=self.groups, axis=get_channel_axis(self.data_format))
            for xi, convi in zip(xx, self.convs):
                yy.append(convi(xi))
            x = tf.concat(yy, axis=get_channel_axis(self.data_format))
        return x


def conv1x1(in_channels,
            out_channels,
            strides=1,
            groups=1,
            use_bias=False,
            data_format="channels_last",
            **kwargs):
    """
    Convolution 1x1 layer.

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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        **kwargs)


def conv3x3(in_channels,
            out_channels,
            strides=1,
            padding=1,
            dilation=1,
            groups=1,
            use_bias=False,
            data_format="channels_last",
            **kwargs):
    """
    Convolution 3x3 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        **kwargs)


def depthwise_conv3x3(channels,
                      strides,
                      data_format="channels_last",
                      **kwargs):
    """
    Depthwise convolution 3x3 layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=channels,
        use_bias=False,
        data_format=data_format,
        **kwargs)


class ConvBlock(nn.Layer):
    """
    Standard convolution block with Batch normalization and activation.

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
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.activate = (activation is not None)
        self.use_bn = use_bn

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
        if self.use_bn:
            self.bn = GluonBatchNormalization(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  strides=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
                  **kwargs):
    """
    1x1 version of the standard convolution block.

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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def conv3x3_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
                  **kwargs):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
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
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def conv5x5_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=2,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
                  **kwargs):
    """
    5x5 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 2
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def conv7x7_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
                  **kwargs):
    """
    7x7 version of the standard convolution block.

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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def dwconv3x3_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation="relu",
                    data_format="channels_last",
                    **kwargs):
    """
    3x3 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return conv3x3_block(
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def dwconv5x5_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=2,
                    dilation=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation="relu",
                    data_format="channels_last",
                    **kwargs):
    """
    5x5 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 2
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return conv5x5_block(
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


class PreConvBlock(nn.Layer):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

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
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
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
                 return_preact=False,
                 activate=True,
                 data_format="channels_last",
                 **kwargs):
        super(PreConvBlock, self).__init__(**kwargs)
        self.return_preact = return_preact
        self.activate = activate

        self.bn = GluonBatchNormalization(
            data_format=data_format,
            name="bn")
        if self.activate:
            self.activ = nn.ReLU()
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

    def call(self, x, training=None):
        x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv1x1_block(in_channels,
                      out_channels,
                      strides=1,
                      use_bias=False,
                      return_preact=False,
                      activate=True,
                      data_format="channels_last",
                      **kwargs):
    """
    1x1 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        use_bias=use_bias,
        return_preact=return_preact,
        activate=activate,
        data_format=data_format,
        **kwargs)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      return_preact=False,
                      activate=True,
                      data_format="channels_last",
                      **kwargs):
    """
    3x3 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        return_preact=return_preact,
        activate=activate,
        data_format=data_format,
        **kwargs)


def channel_shuffle(x,
                    groups,
                    data_format):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.
    data_format : str
        The ordering of the dimensions in tensors.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x_shape = x.get_shape().as_list()
    if is_channels_first(data_format):
        channels = x_shape[1]
        height = x_shape[2]
        width = x_shape[3]
    else:
        height = x_shape[1]
        width = x_shape[2]
        channels = x_shape[3]

    assert (channels % groups == 0)
    channels_per_group = channels // groups

    if is_channels_first(data_format):
        x = tf.reshape(x, shape=(-1, groups, channels_per_group, height, width))
        x = tf.transpose(x, perm=(0, 2, 1, 3, 4))
        x = tf.reshape(x, shape=(-1, channels, height, width))
    else:
        x = tf.reshape(x, shape=(-1, height, width, groups, channels_per_group))
        x = tf.transpose(x, perm=(0, 1, 2, 4, 3))
        x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class ChannelShuffle(nn.Layer):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 groups,
                 data_format="channels_last",
                 **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.groups = groups
        self.data_format = data_format

    def call(self, x):
        return channel_shuffle(x, groups=self.groups, data_format=self.data_format)


def channel_shuffle2(x,
                     channels_per_group,
                     data_format):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.
    The alternative version.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    channels_per_group : int
        Number of groups.
    data_format : str
        Number of channels per group.

    Returns
    -------
    keras.Tensor
        Resulted tensor.
    """
    x_shape = x.get_shape().as_list()
    if is_channels_first(data_format):
        channels = x_shape[1]
        height = x_shape[2]
        width = x_shape[3]
    else:
        height = x_shape[1]
        width = x_shape[2]
        channels = x_shape[3]

    assert (channels % channels_per_group == 0)
    groups = channels // channels_per_group

    if is_channels_first(data_format):
        x = tf.reshape(x, shape=(-1, channels_per_group, groups, height, width))
        x = tf.transpose(x, perm=(0, 2, 1, 3, 4))
        x = tf.reshape(x, shape=(-1, channels, height, width))
    else:
        x = tf.reshape(x, shape=(-1, height, width, channels_per_group, groups))
        x = tf.transpose(x, perm=(0, 1, 2, 4, 3))
        x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class ChannelShuffle2(nn.Layer):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 groups,
                 data_format="channels_last",
                 **kwargs):
        super(ChannelShuffle2, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.channels_per_group = channels // groups
        self.data_format = data_format

    def call(self, x):
        return channel_shuffle2(x, channels_per_group=self.channels_per_group, data_format=self.data_format)


class SEBlock(nn.Layer):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    activation : function, or str, or nn.Layer
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 approx_sigmoid=False,
                 round_mid=False,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.conv1 = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            use_bias=True,
            data_format=data_format,
            name="conv1")
        self.activ = get_activation_layer(activation)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=channels,
            use_bias=True,
            data_format=data_format,
            name="conv2")
        self.sigmoid = HSigmoid() if approx_sigmoid else tf.nn.sigmoid

    def call(self, x, training=None):
        w = self.pool(x)
        axis = -1 if is_channels_first(self.data_format) else 1
        w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x
