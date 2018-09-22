"""
    Common routines for models in Keras.
"""

__all__ = ['conv1x1', 'se_block', 'GluonBatchNormalization']

from keras.backend.mxnet_backend import keras_mxnet_symbol, KerasSymbol
from keras.layers import BatchNormalization
from keras import backend as K
from keras import layers as nn
import mxnet as mx


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
        Block name.
    """
    return nn.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=strides,
        use_bias=use_bias,
        name=name + "/conv")


def se_block(x,
             channels,
             reduction=16,
             name="se_block"):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    name : str, default 'se_block'
        Block name.
    """
    mid_cannels = channels // reduction

    conv1 = conv1x1(
        out_channels=mid_cannels,
        use_bias=True,
        name=name + "/conv1")
    relu = nn.Activation('relu')
    conv2 = conv1x1(
        out_channels=channels,
        use_bias=True,
        name=name + "/conv2")
    sigmoid = nn.Activation('sigmoid')

    assert(len(x.shape) == 4)
    pool_size = x.shape[2:4] if K.image_data_format() == 'channels_first' else x.shape[1:3]
    w = nn.AvgPool2D(pool_size=pool_size)(x)
    w = conv1(w)
    w = relu(w)
    w = conv2(w)
    w = sigmoid(w)
    x = nn.multiply([x, w])
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
