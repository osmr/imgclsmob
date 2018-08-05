"""
    ShuffleNet, implemented in Gluon.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'
"""

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


class ShuffleUnit(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 downsample,
                 ignore_group,
                 **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        with self.name_scope():
            self.compress_conv1 = group_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=(1 if ignore_group else groups))
            self.compress_bn1 = nn.BatchNorm(in_channels=mid_channels)
            self.c_shuffle = ChannelShuffle(
                channels=mid_channels,
                groups=(1 if ignore_group else groups))
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
                self.avgpool = nn.AvgPool2D(
                    pool_size=3,
                    strides=2,
                    padding=1)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        identity = x
        x = self.activ(self.compress_bn1(self.compress_conv1(x)))
        x = self.c_shuffle(x)
        x = self.dw_bn2(self.dw_conv2(x))
        x = self.expand_bn3(self.expand_conv3(x))
        if self.downsample:
            identity = self.avgpool(identity)
            x = F.concat(x, identity, dim=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


class ShuffleInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(ShuffleInitBlock, self).__init__(**kwargs)
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


class ShuffleNet(HybridBlock):

    def __init__(self,
                 channels,
                 init_block_channels,
                 groups,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(ShuffleInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        downsample = (j == 0)
                        ignore_group = (i == 0) and (j == 0)
                        stage.add(ShuffleUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
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


def get_shufflenet(groups,
                   width_scale,
                   pretrained=False,
                   ctx=cpu(),
                   **kwargs):
    init_block_channels = 24
    layers = [4, 8, 4]

    if groups == 1:
        channels_per_layers = [144, 288, 576]
    elif groups == 2:
        channels_per_layers = [200, 400, 800]
    elif groups == 3:
        channels_per_layers = [240, 480, 960]
    elif groups == 4:
        channels_per_layers = [272, 544, 1088]
    elif groups == 8:
        channels_per_layers = [384, 768, 1536]
    else:
        raise ValueError("The {} of groups is not supported".format(groups))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return ShuffleNet(
        channels=channels,
        init_block_channels=init_block_channels,
        groups=groups,
        **kwargs)


def shufflenet_g1_w1(**kwargs):
    return get_shufflenet(1, 1.0, **kwargs)


def shufflenet_g2_w1(**kwargs):
    return get_shufflenet(2, 1.0, **kwargs)


def shufflenet_g3_w1(**kwargs):
    return get_shufflenet(3, 1.0, **kwargs)


def shufflenet_g4_w1(**kwargs):
    return get_shufflenet(4, 1.0, **kwargs)


def shufflenet_g8_w1(**kwargs):
    return get_shufflenet(8, 1.0, **kwargs)


def shufflenet_g1_w3d4(**kwargs):
    return get_shufflenet(1, 0.75, **kwargs)


def shufflenet_g3_w3d4(**kwargs):
    return get_shufflenet(3, 0.75, **kwargs)


def shufflenet_g1_wd2(**kwargs):
    return get_shufflenet(1, 0.5, **kwargs)


def shufflenet_g3_wd2(**kwargs):
    return get_shufflenet(3, 0.5, **kwargs)


def shufflenet_g1_wd4(**kwargs):
    return get_shufflenet(1, 0.25, **kwargs)


def shufflenet_g3_wd4(**kwargs):
    return get_shufflenet(3, 0.25, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    models = [
        shufflenet_g1_w1,
        shufflenet_g2_w1,
        shufflenet_g3_w1,
        shufflenet_g4_w1,
        shufflenet_g8_w1,
        shufflenet_g1_w3d4,
        shufflenet_g3_w3d4,
        shufflenet_g1_wd2,
        shufflenet_g3_wd2,
        shufflenet_g1_wd4,
        shufflenet_g3_wd4,
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
        assert (model != shufflenet_g1_w1 or weight_count == 1531936)
        assert (model != shufflenet_g2_w1 or weight_count == 1733848)
        assert (model != shufflenet_g3_w1 or weight_count == 1865728)
        assert (model != shufflenet_g4_w1 or weight_count == 1968344)
        assert (model != shufflenet_g8_w1 or weight_count == 2434768)
        assert (model != shufflenet_g1_w3d4 or weight_count == 975214)
        assert (model != shufflenet_g3_w3d4 or weight_count == 1238266)
        assert (model != shufflenet_g1_wd2 or weight_count == 534484)
        assert (model != shufflenet_g3_wd2 or weight_count == 718324)
        assert (model != shufflenet_g1_wd4 or weight_count == 209746)
        assert (model != shufflenet_g3_wd4 or weight_count == 305902)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

