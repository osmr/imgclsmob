"""
    Common routines for models in TensorFlow.
"""

__all__ = ['conv2d', 'conv1x1', 'batchnorm', 'maxpool2d', 'se_block', 'channel_shuffle']

import math
import tensorflow as tf


def conv2d(x,
           in_channels,
           out_channels,
           kernel_size,
           strides=1,
           padding=0,
           groups=1,
           use_bias=True,
           name="conv2d"):
    """
    Convolution 2D layer wrapper.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
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
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    name : str, default 'conv2d'
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    if (padding[0] > 0) or (padding[1] > 0):
        x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)])

    if groups == 1:
        x = tf.layers.conv2d(
            inputs=x,
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_first',
            use_bias=use_bias,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
            name=name)
    elif (groups == out_channels) and (out_channels == in_channels):
        kernel = tf.get_variable(
            name=name + '/dw_kernel',
            shape=kernel_size + (in_channels, 1),
            initializer=tf.variance_scaling_initializer(2.0))
        x = tf.nn.depthwise_conv2d(
            input=x,
            filter=kernel,
            strides=(1, 1) + strides,
            padding='VALID',
            rate=(1, 1),
            name=name,
            data_format='NCHW')
        if use_bias:
            raise NotImplementedError
    else:
        assert (in_channels % groups == 0)
        assert (out_channels % groups == 0)
        in_group_channels = in_channels // groups
        out_group_channels = out_channels // groups
        group_list = []
        for gi in range(groups):
            xi = x[:, gi * in_group_channels:(gi + 1) * in_group_channels, :, :]
            xi = tf.layers.conv2d(
                inputs=xi,
                filters=out_group_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding='valid',
                data_format='channels_first',
                use_bias=use_bias,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
                name=name + "/convgroup{}".format(gi + 1))
            group_list.append(xi)
        x = tf.concat(group_list, axis=1, name=name + "/concat")

    return x


def conv1x1(x,
            in_channels,
            out_channels,
            strides=1,
            use_bias=False,
            name="conv1x1"):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    name : str, default 'conv1x1'
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        use_bias=use_bias,
        name=name + "/conv")


def batchnorm(x,
              momentum=0.9,
              epsilon=1e-5,
              training=False,
              name=None):
    """
    Batch normalization layer.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'conv2d'
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = tf.layers.batch_normalization(
        inputs=x,
        axis=1,
        momentum=momentum,
        epsilon=epsilon,
        training=training,
        name=name)
    return x


def maxpool2d(x,
              pool_size,
              strides,
              padding=0,
              ceil_mode=False,
              name=None):
    """
    Max pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the pooling.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    name : str, default 'conv2d'
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    if ceil_mode:
        height = int(x.shape[2])
        out_height = float(height + 2 * padding[0] - pool_size[0]) / strides[0] + 1.0
        if math.ceil(out_height) > math.floor(out_height):
            padding = (padding[0] + 1, padding[1])
        width = int(x.shape[3])
        out_width = float(width + 2 * padding[1] - pool_size[1]) / strides[1] + 1.0
        if math.ceil(out_width) > math.floor(out_width):
            padding = (padding[0], padding[1] + 1)

    if (padding[0] > 0) or (padding[1] > 0):
        x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)], mode="REFLECT")

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=pool_size,
        strides=strides,
        padding='valid',
        data_format='channels_first',
        name=name)
    return x


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    keras.Tensor
        Resulted tensor.
    """
    x_shape = x.get_shape().as_list()
    channels = x_shape[1]
    height = x_shape[2]
    width = x_shape[3]

    assert (channels % groups == 0)
    channels_per_group = channels // groups

    x = tf.reshape(x, shape=(-1, groups, channels_per_group, height, width))
    x = tf.transpose(x, perm=(0, 2, 1, 3, 4))
    x = tf.reshape(x, shape=(-1, channels, height, width))
    return x


def se_block(x,
             channels,
             reduction=16,
             name="se_block"):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    name : str, default 'se_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    assert(len(x.shape) == 4)
    mid_cannels = channels // reduction
    pool_size = x.shape[2:4]

    w = tf.layers.average_pooling2d(
        inputs=x,
        pool_size=pool_size,
        strides=1,
        data_format='channels_first',
        name=name + "/pool")
    w = conv1x1(
        x=w,
        in_channels=channels,
        out_channels=mid_cannels,
        use_bias=True,
        name=name + "/conv1")
    w = tf.nn.relu(w, name=name + "relu")
    w = conv1x1(
        x=w,
        in_channels=mid_cannels,
        out_channels=channels,
        use_bias=True,
        name=name + "/conv2")
    w = tf.nn.sigmoid(w, name=name + "sigmoid")
    x = x * w
    return x
