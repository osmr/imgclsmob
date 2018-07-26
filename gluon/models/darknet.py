"""
    DarkNet, implemented in Gluon.
    Original paper: 'Darknet: Open source neural networks in c'
"""

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .shufflenet import ShuffleInitBlock, ChannelShuffle, depthwise_conv3x3, group_conv1x1


TESTING = False


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


class MEModule(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 side_channels,
                 groups,
                 downsample,
                 ignore_group,
                 **kwargs):
        super(MEModule, self).__init__(**kwargs)
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
        x = self.activ(self.compress_bn1(self.compress_conv1(x)))
        x = self.c_shuffle(x)
        # merging
        y = self.s_merge_conv(x)
        y = self.s_merge_bn(y)
        y = self.activ(y)
        # depthwise convolution (bottleneck)
        x = self.dw_bn2(self.dw_conv2(x))
        # evolution
        y = self.s_conv(y)
        y = self.s_conv_bn(y)
        y = self.activ(y)
        y = self.s_evolve_conv(y)
        y = self.s_evolve_bn(y)
        y = F.sigmoid(y)
        x = x * y
        # pointwise group convolution 2
        x = self.expand_bn3(self.expand_conv3(x))
        # identity branch
        if self.downsample:
            identity = self.avgpool(identity)
            x = F.concat(x, identity, dim=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


class DarkNet(HybridBlock):

    def __init__(self,
                 classes=1000,
                 **kwargs):
        super(DarkNet, self).__init__(**kwargs)
        input_channels = 3
        block_layers = [4, 8, 4]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(ShuffleInitBlock(
                in_channels=input_channels,
                out_channels=block_channels[0]))

            for i in range(len(block_channels) - 1):
                stage = nn.HybridSequential(prefix='')
                in_channels_i = block_channels[i]
                out_channels_i = block_channels[i + 1]
                for j in range(block_layers[i]):
                    stage.add(MEModule(
                        in_channels=(in_channels_i if j == 0 else out_channels_i),
                        out_channels=out_channels_i,
                        side_channels=side_channels,
                        groups=groups,
                        downsample=(j == 0),
                        ignore_group=(i == 0 and j == 0)))
                self.features.add(stage)

            self.features.add(nn.AvgPool2D(pool_size=7))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(
                units=classes,
                in_units=block_channels[-1])

    def hybrid_forward(self, F, x):
        assert ((not TESTING) or x.shape == (1, 3, 224, 224))
        assert ((not TESTING) or self.features[0:1](x).shape == (1, 12, 56, 56))
        assert ((not TESTING) or self.features[0:2](x).shape == (1, 108, 28, 28))

        x = self.features(x)
        #assert ((not TESTING) or x.shape == (1, 432, 7, 7))
        x = self.output(x)
        assert ((not TESTING) or x.shape == (1, 1000))

        return x


def get_darknet(pretrained=False,
                ctx=cpu(),
                **kwargs):
    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = DarkNet(**kwargs)
    return net


def darknet(**kwargs):
    return get_darknet(**kwargs)

