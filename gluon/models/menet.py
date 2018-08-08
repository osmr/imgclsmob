"""
    MENet, implemented in Gluon.
    Original paper: 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications,'
    https://arxiv.org/abs/1803.09127.
"""

__all__ = ['MENet', 'menet108_8x1_g3', 'menet128_8x1_g4', 'menet160_8x1_g8', 'menet228_12x1_g3', 'menet256_12x1_g4',
           'menet348_12x1_g3', 'menet352_12x1_g8', 'menet456_24x1_g3']

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


TESTING = False


def depthwise_conv3x3(channels,
                      strides):
    return nn.Conv2D(
        channels=channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=channels,
        use_bias=False,
        in_channels=channels)


def group_conv1x1(in_channels,
                  out_channels,
                  groups):
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        groups=groups,
        use_bias=False,
        in_channels=in_channels)


def conv1x1(in_channels,
            out_channels):
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        use_bias=False,
        in_channels=in_channels)


def conv3x3(in_channels,
            out_channels,
            strides):
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        use_bias=False,
        in_channels=in_channels)


def channel_shuffle(x,
                    groups):
    return x.reshape((0, -4, groups, -1, -2)).swapaxes(1, 2).reshape((0, -3, -2))


class ChannelShuffle(HybridBlock):

    def __init__(self,
                 channels,
                 groups,
                 **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.groups = groups

    def hybrid_forward(self, F, x):
        return channel_shuffle(x, self.groups)


class MEUnit(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 side_channels,
                 groups,
                 downsample,
                 ignore_group,
                 **kwargs):
        super(MEUnit, self).__init__(**kwargs)
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        with self.name_scope():
            # residual branch
            self.compress_conv1 = group_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=(1 if ignore_group else groups))
            self.compress_bn1 = nn.BatchNorm(in_channels=mid_channels)
            self.c_shuffle = ChannelShuffle(
                channels=mid_channels,
                groups=groups)
            self.dw_conv2 = depthwise_conv3x3(
                channels=mid_channels,
                strides=(2 if self.downsample else 1))
            self.dw_bn2 = nn.BatchNorm(in_channels=mid_channels)
            self.expand_conv3 = group_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                groups=groups)
            self.expand_bn3 = nn.BatchNorm(in_channels=out_channels)
            if downsample:
                self.avgpool = nn.AvgPool2D(pool_size=3, strides=2, padding=1)
            self.activ = nn.Activation('relu')

            # fusion branch
            self.s_merge_conv = conv1x1(
                in_channels=mid_channels,
                out_channels=side_channels)
            self.s_merge_bn = nn.BatchNorm(in_channels=side_channels)
            self.s_conv = conv3x3(
                in_channels=side_channels,
                out_channels=side_channels,
                strides=(2 if self.downsample else 1))
            self.s_conv_bn = nn.BatchNorm(in_channels=side_channels)
            self.s_evolve_conv = conv1x1(
                in_channels=side_channels,
                out_channels=mid_channels)
            self.s_evolve_bn = nn.BatchNorm(in_channels=mid_channels)

    def hybrid_forward(self, F, x):
        identity = x
        # pointwise group convolution 1
        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        # merging
        y = self.s_merge_conv(x)
        y = self.s_merge_bn(y)
        y = self.activ(y)
        # depthwise convolution (bottleneck)
        x = self.dw_conv2(x)
        x = self.dw_bn2(x)
        # evolution
        y = self.s_conv(y)
        y = self.s_conv_bn(y)
        y = self.activ(y)
        y = self.s_evolve_conv(y)
        y = self.s_evolve_bn(y)
        y = F.sigmoid(y)
        x = x * y
        # pointwise group convolution 2
        x = self.expand_conv3(x)
        x = self.expand_bn3(x)
        # identity branch
        if self.downsample:
            identity = self.avgpool(identity)
            x = F.concat(x, identity, dim=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


class MEInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(MEInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=3,
                strides=2,
                padding=1,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)
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


class MENet(HybridBlock):
    """
    MENet model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications,'
    https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    side_channels : int
        Number of side channels in a ME-unit.
    groups : int
        Number of groups in convolution layers.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 side_channels,
                 groups,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(MENet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(MEInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        downsample = (j == 0)
                        ignore_group = (i == 0) and (j == 0)
                        stage.add(MEUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            side_channels=side_channels,
                            groups=groups,
                            downsample=downsample,
                            ignore_group=ignore_group))
                        in_channels = out_channels
                self.features.add(stage)
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


def get_menet(first_stage_channels,
              side_channels,
              groups,
              pretrained=False,
              ctx=cpu(),
              **kwargs):
    layers = [4, 8, 4]

    if first_stage_channels == 108:
        init_block_channels = 12
        channels_per_layers = [108, 216, 432]
    elif first_stage_channels == 128:
        init_block_channels = 12
        channels_per_layers = [128, 256, 512]
    elif first_stage_channels == 160:
        init_block_channels = 16
        channels_per_layers = [160, 320, 640]
    elif first_stage_channels == 228:
        init_block_channels = 24
        channels_per_layers = [228, 456, 912]
    elif first_stage_channels == 256:
        init_block_channels = 24
        channels_per_layers = [256, 512, 1024]
    elif first_stage_channels == 348:
        init_block_channels = 24
        channels_per_layers = [348, 696, 1392]
    elif first_stage_channels == 352:
        init_block_channels = 24
        channels_per_layers = [352, 704, 1408]
    elif first_stage_channels == 456:
        init_block_channels = 48
        channels_per_layers = [456, 912, 1824]
    else:
        raise ValueError("The {} of `first_stage_channels` is not supported".format(first_stage_channels))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MENet(
        channels=channels,
        init_block_channels=init_block_channels,
        side_channels=side_channels,
        groups=groups,
        **kwargs)


def menet108_8x1_g3(**kwargs):
    """
    108-MENet-8x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(108, 8, 3, **kwargs)


def menet128_8x1_g4(**kwargs):
    """
    128-MENet-8x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(128, 8, 4, **kwargs)


def menet160_8x1_g8(**kwargs):
    """
    160-MENet-8x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(160, 8, 8, **kwargs)


def menet228_12x1_g3(**kwargs):
    """
    228-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(228, 12, 3, **kwargs)


def menet256_12x1_g4(**kwargs):
    """
    256-MENet-12x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(256, 12, 4, **kwargs)


def menet348_12x1_g3(**kwargs):
    """
    348-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(348, 12, 3, **kwargs)


def menet352_12x1_g8(**kwargs):
    """
    352-MENet-12x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(352, 12, 8, **kwargs)


def menet456_24x1_g3(**kwargs):
    """
    456-MENet-24x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_menet(456, 24, 3, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    models = [
        menet108_8x1_g3,
        menet128_8x1_g4,
        menet160_8x1_g8,
        menet228_12x1_g3,
        menet256_12x1_g4,
        menet348_12x1_g3,
        menet352_12x1_g8,
        menet456_24x1_g3,
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
        assert (model != menet108_8x1_g3 or weight_count == 654516)
        assert (model != menet128_8x1_g4 or weight_count == 750796)
        assert (model != menet160_8x1_g8 or weight_count == 850120)
        assert (model != menet228_12x1_g3 or weight_count == 1806568)
        assert (model != menet256_12x1_g4 or weight_count == 1888240)
        assert (model != menet348_12x1_g3 or weight_count == 3368128)
        assert (model != menet352_12x1_g8 or weight_count == 2272872)
        assert (model != menet456_24x1_g3 or weight_count == 5304784)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

