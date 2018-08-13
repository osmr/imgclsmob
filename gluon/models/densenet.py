"""
    DenseNet, implemented in Gluon.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class DenseConv(HybridBlock):
    """
    DenseNet specific convolution block.

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
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 bn_use_global_stats,
                 **kwargs):
        super(DenseConv, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation('relu')
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def dense_conv1x1(in_channels,
                  out_channels,
                  bn_use_global_stats):
    """
    1x1 version of the DenseNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return DenseConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        bn_use_global_stats=bn_use_global_stats)


def dense_conv3x3(in_channels,
                  out_channels,
                  bn_use_global_stats):
    """
    3x3 version of the DenseNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return DenseConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1,
        bn_use_global_stats=bn_use_global_stats)


class DenseUnit(HybridBlock):
    """
    DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : bool
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 dropout_rate,
                 **kwargs):
        super(DenseUnit, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        with self.name_scope():
            self.conv1 = dense_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = dense_conv3x3(
                in_channels=mid_channels,
                out_channels=inc_channels,
                bn_use_global_stats=bn_use_global_stats)
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.concat(identity, x, dim=1)
        return x


class TransitionBlock(HybridBlock):
    """
    DenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = dense_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.pool = nn.AvgPool2D(
                pool_size=2,
                strides=2,
                padding=0)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseInitBlock(HybridBlock):
    """
    DenseNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(DenseInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=7,
                strides=2,
                padding=3,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation('relu')
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class PostActivation(HybridBlock):
    """
    DenseNet final block, which performs the same function of postactivation as in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(PostActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class DenseNet(HybridBlock):
    """
    DenseNet model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bn_use_global_stats=False,
                 dropout_rate=0.0,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(DenseNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(DenseInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    if i != 0:
                        stage.add(TransitionBlock(
                            in_channels=in_channels,
                            out_channels=(in_channels // 2),
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = in_channels // 2
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(DenseUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bn_use_global_stats=bn_use_global_stats,
                            dropout_rate=dropout_rate))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PostActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
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


def get_densenet(num_layers,
                 pretrained=False,
                 ctx=cpu(),
                 **kwargs):
    """
    Create DenseNet model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """

    if num_layers == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif num_layers == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif num_layers == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif num_layers == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported DenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(lambda xi, yi:
                      xi + [reduce(lambda xj, yj:
                                   xj + [xj[-1] + yj],
                                   [growth_rate] * yi,
                                   [xi[-1][-1] // 2])[1:]],
                      layers,
                      [[init_block_channels * 2]])[1:]

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return DenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)


def densenet121(**kwargs):
    """
    DenseNet-121 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_densenet(121, **kwargs)


def densenet161(**kwargs):
    """
    DenseNet-161 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_densenet(161, **kwargs)


def densenet169(**kwargs):
    """
    DenseNet-169 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_densenet(169, **kwargs)


def densenet201(**kwargs):
    """
    DenseNet-201 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_densenet(201, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    models = [
        densenet121,
        densenet161,
        densenet169,
        densenet201,
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
        assert (model != densenet121 or weight_count == 7978856)
        assert (model != densenet161 or weight_count == 28681000)
        assert (model != densenet169 or weight_count == 14149480)
        assert (model != densenet201 or weight_count == 20013928)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

