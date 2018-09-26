"""
    Common routines for models in Keras.
"""

__all__ = ['conv2d', 'conv1x1', 'max_pool2d_ceil', 'channel_shuffle_lambda', 'se_block', 'GluonBatchNormalization']

import math
from keras.backend.mxnet_backend import keras_mxnet_symbol, KerasSymbol
from keras.layers import BatchNormalization
from keras import backend as K
from keras import layers as nn
import mxnet as mx


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
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    extra_pad = False
    if (padding[0] == padding[1]) and (padding[0] == 0):
        ke_padding = "valid"
    elif (padding[0] == padding[1]) and (kernel_size[0] == kernel_size[1]) and (kernel_size[0] // 2 == padding[0]):
        ke_padding = "same"
    else:
        x = nn.ZeroPadding2D(
            padding=padding,
            name=name + "/pad")(x)
        ke_padding = "valid"
        extra_pad = True

    if groups == 1:
        if extra_pad:
            name = name + "/conv"
        x = nn.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=ke_padding,
            use_bias=use_bias,
            name=name)(x)
    elif (groups == out_channels) and (out_channels == in_channels):
        if extra_pad:
            name = name + "/conv"
        x = nn.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=ke_padding,
            use_bias=use_bias,
            name=name)(x)
    else:
        assert (in_channels % groups == 0)
        assert (out_channels % groups == 0)
        in_group_channels = in_channels // groups
        out_group_channels = out_channels // groups
        group_list = []
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        for gi in range(groups):
            xi = nn.Lambda(
                lambda z: z[:, gi * in_group_channels:(gi + 1) * in_group_channels, :, :]
                if K.image_data_format() == 'channels_first' else
                lambda z: z[:, :, :, gi * in_group_channels:(gi + 1) * in_group_channels])(x)
            xi = nn.Conv2D(
                filters=out_group_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=ke_padding,
                use_bias=use_bias,
                name=name + "/convgroup{}".format(gi + 1))(xi)
            group_list.append(xi)
        x = nn.concatenate(group_list, axis=channel_axis, name=name + "/concat")

    return x


def conv1x1(out_channels,
            strides=1,
            use_bias=False,
            name="conv1x1"):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    name : str, default 'conv1x1'
        Layer name.
    """
    return nn.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=strides,
        use_bias=use_bias,
        name=name + "/conv")


def max_pool2d_ceil(x,
                    pool_size,
                    strides,
                    padding,
                    **kwargs):
    """
    Max pooling 2D layer wrapper with ceil mode supporting.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the pooling.
    padding : int or tuple/list of 2 int
        Padding value for pooling layer.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)

    assert (pool_size[0] == pool_size[1])
    assert (strides[0] == strides[1])

    padding0 = 0 if padding == "valid" else strides[0] // 2

    height = x.shape[2 if K.image_data_format() == 'channels_first' else 1]
    out_height = float(height + 2 * padding0 - pool_size[0]) / strides[0] + 1.0
    if math.ceil(out_height) > math.floor(out_height):
        assert (strides[0] <= 3)
        padding = "same"

    x = nn.MaxPool2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        **kwargs)(x)
    return x


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    groups : int
        Number of groups.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """

    is_channels_first = (K.image_data_format() == 'channels_first')
    if is_channels_first:
        batch, channels, height, width = x.shape
    else:
        batch, height, width, channels = x.shape

    # assert (channels % groups == 0)
    channels_per_group = channels // groups

    if is_channels_first:
        x = K.reshape(x, shape=(-1, groups, channels_per_group, height, width))
        x = K.permute_dimensions(x, pattern=(0, 2, 1, 3, 4))
        x = K.reshape(x, shape=(-1, channels, height, width))
    else:
        x = K.reshape(x, shape=(batch, height, width, groups, channels_per_group))
        x = K.permute_dimensions(x, pattern=(0, 1, 2, 4, 3))
        x = K.reshape(x, shape=(batch, height, width, channels))

    if not hasattr(x, '_keras_shape'):
        x._keras_shape = tuple([d if d != 0 else None for d in x.shape])
    return x


def channel_shuffle_lambda(channels,
                           groups,
                           **kwargs):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.

    Returns
    -------
    Layer
        Channel shuffle layer.
    """
    assert (channels % groups == 0)

    return nn.Lambda(channel_shuffle, arguments={'groups': groups}, **kwargs)


def se_block(x,
             channels,
             reduction=16,
             name="se_block"):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    name : str, default 'se_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_cannels = channels // reduction

    conv1 = conv1x1(
        out_channels=mid_cannels,
        use_bias=True,
        name=name + "/conv1")
    relu = nn.Activation('relu', name=name + "/relu")
    conv2 = conv1x1(
        out_channels=channels,
        use_bias=True,
        name=name + "/conv2")
    sigmoid = nn.Activation('sigmoid', name=name + "/sigmoid")

    assert(len(x.shape) == 4)
    pool_size = x.shape[2:4] if K.image_data_format() == 'channels_first' else x.shape[1:3]
    w = nn.AvgPool2D(
        pool_size=pool_size,
        name=name + "/pool")(x)
    w = conv1(w)
    w = relu(w)
    w = conv2(w)
    w = sigmoid(w)
    x = nn.multiply([x, w], name=name + "/mul")
    return x


@keras_mxnet_symbol
def gluon_batchnorm(x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    momentum=0.9,
                    axis=1,
                    epsilon=1e-5,
                    fix_gamma=False):
    """
    Apply native MXNet/Gluon batch normalization on x with given moving_mean, moving_var, beta and gamma.


    Parameters
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    gamma : keras.backend tensor/variable/symbol
        Tensor by which to scale the input.
    beta : keras.backend tensor/variable/symbol
        Tensor by which to center the input.
    moving_mean : keras.backend tensor/variable/symbol
        Moving mean.
    moving_var : keras.backend tensor/variable/symbol
        Moving variance.
    momentum : float, default 0.9
        Momentum for the moving average.
    axis : int, default 1
        Axis along which BatchNorm is applied. Axis usually represent axis of 'channels'. MXNet follows
        'channels_first'.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    fix_gamma : bool, default False
        Fix gamma while training.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(moving_mean, KerasSymbol):
        moving_mean = moving_mean.symbol
    if isinstance(moving_var, KerasSymbol):
        moving_var = moving_var.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol
    return KerasSymbol(mx.sym.BatchNorm(
        data=x,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        momentum=momentum,
        axis=axis,
        eps=epsilon,
        fix_gamma=fix_gamma))


class GluonBatchNormalization(BatchNormalization):
    """
    Batch normalization layer wrapper for implementation of the Gluon type of BatchNorm default parameters.

    Parameters
    ----------
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center : bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale : bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer : str, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer : str, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer : str, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer : str, default 'ones'
        Initializer for the moving variance.
    beta_regularizer : str or None, default None
        Optional regularizer for the beta weight.
    gamma_regularizer : str or None, default None
        Optional regularizer for the gamma weight.
    beta_constraint : str or None, default None
        Optional constraint for the beta weight.
    gamma_constraint : str or None, default None
        Optional constraint for the gamma weight.
    fix_gamma : bool, default False
        Fix gamma while training.
    """
    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 fix_gamma=False,
                 **kwargs):
        super(GluonBatchNormalization, self).__init__(
            axis=(1 if K.image_data_format() == 'channels_first' else 3),
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs)
        self.fix_gamma = fix_gamma

    def call(self, inputs, training=None):
        if K.backend() == 'mxnet':
            return gluon_batchnorm(
                x=inputs,
                gamma=self.gamma,
                beta=self.beta,
                moving_mean=self.moving_mean,
                moving_var=self.moving_variance,
                momentum=self.momentum,
                axis=self.axis,
                epsilon=self.epsilon,
                fix_gamma=self.fix_gamma)
        else:
            super(GluonBatchNormalization, self).call(inputs, training)
