"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['oth_cifar_resnext29_32x4d', 'oth_cifar_resnext29_16x64d']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock


class CIFARBlock(HybridBlock):
    r"""Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channels, cardinality, bottleneck_width,
                 stride, downsample=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(CIFARBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(group_width, kernel_size=1, use_bias=False))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(group_width, kernel_size=3, strides=stride, padding=1,
                                groups=cardinality, use_bias=False))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels * 4, kernel_size=1, use_bias=False))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x

# Nets
class CIFARResNext(HybridBlock):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 10
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, layers, cardinality, bottleneck_width, classes=10,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(CIFARResNext, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels = 64

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels, 3, 1, 1, use_bias=False))
            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(channels, num_layer, stride, i+1,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                channels *= 2
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes)

    def _make_layer(self, channels, num_layer, stride, stage_index,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(CIFARBlock(channels, self.cardinality, self.bottleneck_width,
                                 stride, True, prefix='',
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(num_layer-1):
                layer.add(CIFARBlock(channels, self.cardinality, self.bottleneck_width,
                                     1, False, prefix='',
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


# Constructor
def get_cifar_resnext(num_layers, cardinality=16, bottleneck_width=64, in_channels=3,
                      pretrained=False, ctx=cpu(),
                      root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 9*n+2, e.g. 29
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    assert (num_layers - 2) % 9 == 0
    layer = (num_layers - 2) // 9
    layers = [layer] * 3
    net = CIFARResNext(layers, cardinality, bottleneck_width, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('cifar_resnext%d_%dx%dd'%(num_layers, cardinality,
                                                                     bottleneck_width),
                                           tag=pretrained, root=root), ctx=ctx)
    return net

def oth_cifar_resnext29_32x4d(**kwargs):
    r"""ResNext-29 32x4d model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 9*n+2, e.g. 29
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_cifar_resnext(29, 32, 4, **kwargs)

def oth_cifar_resnext29_16x64d(**kwargs):
    r"""ResNext-29 16x64d model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_cifar_resnext(29, 16, 64, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        oth_cifar_resnext29_32x4d,
        oth_cifar_resnext29_16x64d,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 10))

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_cifar_resnext29_32x4d or weight_count == 4775754)
        assert (model != oth_cifar_resnext29_16x64d or weight_count == 68155210)


if __name__ == "__main__":
    _test()
