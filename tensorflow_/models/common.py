"""
    Common routines for models in TensorFlow.
"""

__all__ = ['conv2d', 'conv1x1', 'se_block', 'channel_shuffle']

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

    if (padding[0] == padding[1]) and (padding[0] == 0):
        tf_padding = "valid"
    elif (padding[0] == padding[1]) and (kernel_size[0] == kernel_size[1]) and (kernel_size[0] // 2 == padding[0]):
        tf_padding = "same"
    else:
        raise NotImplementedError

    if groups != 1:
        raise NotImplementedError

    x = tf.layers.conv2d(
        inputs=x,
        filters=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=tf_padding,
        data_format='channels_first',
        use_bias=use_bias,
        name=name)

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
    assert channels % groups == 0, channels
    x = tf.reshape(x, [-1, channels // groups, groups] + x_shape[-2:])
    x = tf.transpose(x, [0, 2, 1, 3, 4])
    x = tf.reshape(x, [-1, channels] + x_shape[-2:])
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
