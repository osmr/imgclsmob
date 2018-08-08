"""
    SqueezeNext, implemented in Gluon.
    Original paper: 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
"""

__all__ = ['SqueezeNext', 'sqnxt23_w1', 'sqnxt23_w3d2', 'sqnxt23_w2', 'sqnxt23v5_w1', 'sqnxt23v5_w3d2', 'sqnxt23v5_w2']

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class SqnxtConv(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding=(0, 0),
                 **kwargs):
        super(SqnxtConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class SqnxtUnit(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(SqnxtUnit, self).__init__(**kwargs)
        if strides == 2:
            reduction_den = 1
            self.resize_identity = True
        elif in_channels > out_channels:
            reduction_den = 4
            self.resize_identity = True
        else:
            reduction_den = 2
            self.resize_identity = False

        with self.name_scope():
            self.conv1 = SqnxtConv(
                in_channels=in_channels,
                out_channels=(in_channels // reduction_den),
                kernel_size=1,
                strides=strides)
            self.conv2 = SqnxtConv(
                in_channels=(in_channels // reduction_den),
                out_channels=(in_channels // (2 * reduction_den)),
                kernel_size=1,
                strides=1)
            self.conv3 = SqnxtConv(
                in_channels=(in_channels // (2 * reduction_den)),
                out_channels=(in_channels // reduction_den),
                kernel_size=(1, 3),
                strides=1,
                padding=(0, 1))
            self.conv4 = SqnxtConv(
                in_channels=(in_channels // reduction_den),
                out_channels=(in_channels // reduction_den),
                kernel_size=(3, 1),
                strides=1,
                padding=(1, 0))
            self.conv5 = SqnxtConv(
                in_channels=(in_channels // reduction_den),
                out_channels=out_channels,
                kernel_size=1,
                strides=1)

            if self.resize_identity:
                self.identity_conv = SqnxtConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    strides=strides)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        identity = self.activ(identity)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + identity
        x = self.activ(x)
        return x


class SqnxtInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(SqnxtInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = SqnxtConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                strides=2,
                padding=1)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                ceil_mode=True)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class SqueezeNext(HybridBlock):
    """
    SqueezeNext model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(SqueezeNext, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(SqnxtInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(SqnxtUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(SqnxtConv(
                in_channels=in_channels,
                out_channels=final_block_channels,
                kernel_size=1,
                strides=1))
            in_channels = final_block_channels
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_squeezenext(version,
                    width_scale,
                    pretrained=False,
                    ctx=cpu(),
                    **kwargs):
    init_block_channels = 64
    final_block_channels = 128
    channels_per_layers = [32, 64, 128, 256]

    if version == '23':
        layers = [6, 6, 8, 1]
    elif version == '23v5':
        layers = [2, 4, 14, 1]
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        final_block_channels = int(final_block_channels * width_scale)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return SqueezeNext(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)


def sqnxt23_w1(**kwargs):
    """
    1.0-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_squeezenext('23', 1.0, **kwargs)


def sqnxt23_w3d2(**kwargs):
    """
    0.75-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_squeezenext('23', 1.5, **kwargs)


def sqnxt23_w2(**kwargs):
    """
    0.5-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_squeezenext('23', 2.0, **kwargs)


def sqnxt23v5_w1(**kwargs):
    """
    1.0-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_squeezenext('23v5', 1.0, **kwargs)


def sqnxt23v5_w3d2(**kwargs):
    """
    0.75-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_squeezenext('23v5', 1.5, **kwargs)


def sqnxt23v5_w2(**kwargs):
    """
    0.5-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_squeezenext('23v5', 2.0, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    models = [
        sqnxt23_w1,
        sqnxt23_w3d2,
        sqnxt23_w2,
        sqnxt23v5_w1,
        sqnxt23v5_w3d2,
        sqnxt23v5_w2,
    ]

    for model in models:

        net = model()

        ctx = mx.cpu()
        net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        assert (model != sqnxt23_w1 or weight_count == 724056)
        assert (model != sqnxt23_w3d2 or weight_count == 1511824)
        assert (model != sqnxt23_w2 or weight_count == 2583752)
        assert (model != sqnxt23v5_w1 or weight_count == 921816)
        assert (model != sqnxt23v5_w3d2 or weight_count == 1953616)
        assert (model != sqnxt23v5_w2 or weight_count == 3366344)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

