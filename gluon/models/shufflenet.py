"""
    ShuffleNet, implemented in Gluon.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'
"""

import numpy as np
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
        # with self.name_scope():
        #     pass

    def hybrid_forward(self, F, x):
        return channel_shuffle(x, self.groups)


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
                self.avgpool = nn.AvgPool2D(pool_size=3, strides=2, padding=1)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        identity = x
        out = self.activ(self.compress_bn1(self.compress_conv1(x)))
        out = self.c_shuffle(out)
        out = self.dw_bn2(self.dw_conv2(out))
        out = self.expand_bn3(self.expand_conv3(out))
        if self.downsample:
            identity = self.avgpool(identity)
            out = F.concat(out, identity, dim=1)
        else:
            out = out + identity
        out = self.activ(out)
        return out


class ShuffleNet(HybridBlock):

    def __init__(self,
                 groups,
                 stage_out_channels,
                 classes=1000,
                 **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)
        stage_num_blocks = [4, 8, 4]
        input_channels = 3

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(ShuffleInitBlock(
                in_channels=input_channels,
                out_channels=stage_out_channels[0]))

            for i in range(len(stage_num_blocks)):
                stage = nn.HybridSequential(prefix='')
                in_channels_i = stage_out_channels[i]
                out_channels_i = stage_out_channels[i + 1]
                for j in range(stage_num_blocks[i]):
                    stage.add(ShuffleUnit(
                        in_channels=(in_channels_i if j == 0 else out_channels_i),
                        out_channels=out_channels_i,
                        groups=groups,
                        downsample=(j == 0),
                        ignore_group=(i == 0 and j == 0)))
                self.features.add(stage)

            self.features.add(nn.AvgPool2D(pool_size=7))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(
                units=classes,
                in_units=stage_out_channels[-1])

    def hybrid_forward(self, F, x):
        assert ((not TESTING) or x.shape == (1, 3, 224, 224))

        x = self.features(x)
        assert ((not TESTING) or x.shape == (1, 800))
        x = self.output(x)
        assert ((not TESTING) or x.shape == (1, 1000))

        return x


def get_shufflenet(scale,
                   groups,
                   pretrained=False,
                   ctx=cpu(),
                   **kwargs):
    if groups == 1:
        stage_out_channels = [24, 144, 288, 576]
    elif groups == 2:
        stage_out_channels = [24, 200, 400, 800]
    elif groups == 3:
        stage_out_channels = [24, 240, 480, 960]
    elif groups == 4:
        stage_out_channels = [24, 272, 544, 1088]
    elif groups == 8:
        stage_out_channels = [24, 384, 768, 1536]
    else:
        raise ValueError("The {} of groups is not supported".format(groups))
    stage_out_channels = (np.array(stage_out_channels) * scale).astype(np.int)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = ShuffleNet(
        groups=groups,
        stage_out_channels=stage_out_channels,
        **kwargs)
    return net


def shufflenet1_0_g1(**kwargs):
    return get_shufflenet(1.0, 1, **kwargs)


def shufflenet1_0_g2(**kwargs):
    return get_shufflenet(1.0, 2, **kwargs)


def shufflenet1_0_g3(**kwargs):
    return get_shufflenet(1.0, 3, **kwargs)


def shufflenet1_0_g4(**kwargs):
    return get_shufflenet(1.0, 4, **kwargs)


def shufflenet1_0_g8(**kwargs):
    return get_shufflenet(1.0, 8, **kwargs)


def shufflenet0_5_g1(**kwargs):
    return get_shufflenet(0.5, 1, **kwargs)


# def shufflenet0_5_g2(**kwargs):
#     return get_shufflenet(0.5, 2, **kwargs)


def shufflenet0_5_g3(**kwargs):
    return get_shufflenet(0.5, 3, **kwargs)


# def shufflenet0_5_g4(**kwargs):
#     return get_shufflenet(0.5, 4, **kwargs)


# def shufflenet0_5_g8(**kwargs):
#     return get_shufflenet(0.5, 8, **kwargs)


def shufflenet0_25_g1(**kwargs):
    return get_shufflenet(0.25, 1, **kwargs)


# def shufflenet0_25_g2(**kwargs):
#     return get_shufflenet(0.25, 2, **kwargs)


def shufflenet0_25_g3(**kwargs):
    return get_shufflenet(0.25, 3, **kwargs)


# def shufflenet0_25_g4(**kwargs):
#     return get_shufflenet(0.25, 4, **kwargs)


# def shufflenet0_25_g8(**kwargs):
#     return get_shufflenet(0.25, 8, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    net = shufflenet1_0_g2()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    assert (weight_count == 1733848)

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

