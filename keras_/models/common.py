"""
    Common routines for models in Keras.
"""

__all__ = ['GluonBatchNormalization']

from keras.backend.mxnet_backend import keras_mxnet_symbol, KerasSymbol
from keras.layers import BatchNormalization
from keras import backend as K
import mxnet as mx


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
                 axis=1,
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
            axis=axis,
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
