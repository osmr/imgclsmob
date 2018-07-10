"""
    ShuffleNet, implemented in Gluon.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'
"""

import numpy as np
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class ChannelShuffle(HybridBlock):

    def __init__(self,
                 groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def hybrid_forward(self, F, x):
        N, C, H, W = x.size()
        G = self.groups
        assert(C % G == 0)
        return x.reshape((N, G, C//G, H, W)).swapaxes(1, 2).reshape((N, C, H, W))


class ShuffleBottleneck(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 stride,
                 first):
        super(ShuffleBottleneck, self).__init__()
        self.stride = stride
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = nn.Conv2D(
                channels=mid_channels,
                kernel_size=1,
                groups=(1 if first else groups),
                use_bias=False,
                in_channels=in_channels)
            self.bn1 = nn.BatchNorm(in_channels=mid_channels)
            self.shuffle = ChannelShuffle(groups=(1 if first else groups))
            self.conv2 = nn.Conv2D(
                channels=mid_channels,
                kernel_size=3,
                strides=(2 if self.stride else 1),
                padding=1,
                groups=mid_channels,
                use_bias=False,
                in_channels=mid_channels)
            self.bn2 = nn.BatchNorm(in_channels=mid_channels)
            self.conv3 = nn.Conv2D(
                channels=out_channels,
                kernel_size=1,
                groups=groups,
                use_bias=False,
                in_channels=mid_channels)
            self.bn3 = nn.BatchNorm(in_channels=out_channels)
            self.avgpool = nn.AvgPool2D(pool_size=3, strides=2, padding=1)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        res = self.avgpool(x) if self.stride else x
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.shuffle(out)
        out = self.activ(self.bn2(self.conv2(out)))
        out = self.activ(self.bn3(self.conv3(out)))
        out = self.activ(F.concat((out, res), dim=1)) if self.stride else self.activ(F.elemwise_add(out, res))
        return out


class ShuffleNet(HybridBlock):

    def __init__(self,
                 groups,
                 stage_out_channels,
                 classes=1000,
                 **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)
        num_blocks = [4, 8, 4]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(
                channels=stage_out_channels[0],
                kernel_size=1,
                use_bias=False,
                in_channels=3))
            self.features.add(nn.BatchNorm(in_channels=stage_out_channels[0]))
            self.features.add(nn.Activation('relu'))

            self.features.add(self._make_stage(
                stage_out_channels[0], stage_out_channels[1], num_blocks[0], groups, True))
            self.features.add(self._make_stage(
                stage_out_channels[1], stage_out_channels[2], num_blocks[1], groups, False))
            self.features.add(self._make_stage(
                stage_out_channels[2], stage_out_channels[3], num_blocks[2], groups, False))

            self.features.add(nn.AvgPool2D(pool_size=4))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    num_blocks,
                    groups,
                    first):
        out = nn.HybridSequential(prefix='')
        for i in range(num_blocks):
            out.add(ShuffleBottleneck(
                in_channels=in_channels,
                out_channels=(out_channels - in_channels if i == 0 else out_channels),
                groups=groups,
                stride=(i == 0),
                first=first))
            in_channels = out_channels
        return out

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
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
        raise Exception()
    stage_out_channels = (np.array(stage_out_channels) * scale).astype(np.int)
    net = ShuffleNet(
        groups=groups,
        stage_out_channels=stage_out_channels,
        **kwargs)
    return net


def shufflenet1_0_g1(**kwargs):
    return get_shufflenet(1.0, 1, **kwargs)


def shufflenet1_0_g2(**kwargs):
    return get_shufflenet(1.0, 2, **kwargs)

