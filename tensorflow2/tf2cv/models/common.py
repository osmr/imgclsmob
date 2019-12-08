"""
    Common routines for models in TensorFlow 2.0.
"""

__all__ = ['is_channels_first', 'get_channel_axis', 'flatten', 'MaxPool2d', 'ConvBlock', 'conv1x1_block',
           'conv3x3_block', 'conv7x7_block']

import math
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


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function or str
        Activation function or name of activation function.

    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert (activation is not None)
    if isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        else:
            raise NotImplementedError()
    else:
        return activation()


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
            assert self.use_pad
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
            height = int(x.shape[2])
            out_height = float(height + 2 * self.padding[0] - self.pool_size[0]) / self.strides[0] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (self.padding[0] + 1, self.padding[1])
            width = int(x.shape[3])
            out_width = float(width + 2 * self.padding[1] - self.pool_size[1]) / self.strides[1] + 1.0
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)
            if is_channels_first(self.data_format):
                paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
            else:
                paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]
        elif self.use_pad:
            paddings_tf = self.paddings_tf

        if self.use_pad:
            x = tf.pad(x, paddings=paddings_tf)
        x = self.pool(x)
        return x


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

        assert (groups == 1)

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
            # if is_channels_first(data_format):
            #     self.paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
            # else:
            #     self.paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]
        self.conv = nn.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            data_format=data_format,
            dilation_rate=dilation,
            use_bias=use_bias,
            name="conv")
        if self.use_bn:
            self.bn = GluonBatchNormalization(
                epsilon=bn_eps,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation)

    def call(self, x, training=None):
        if self.use_pad:
            x = self.pad(x)
            # x = tf.pad(x, paddings=self.paddings_tf)
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
