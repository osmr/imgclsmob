"""
    MENet, implemented in Gluon.
    Original paper: 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'
"""

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .shufflenet import ShuffleInitBlock, ChannelShuffle, depthwise_conv3x3, group_conv1x1


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


class MENet(HybridBlock):

    def __init__(self,
                 block_channels,
                 side_channels,
                 groups,
                 classes=1000,
                 **kwargs):
        super(MENet, self).__init__(**kwargs)
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
        x = self.features(x)
        x = self.output(x)
        return x


def get_menet(first_block_channels,
              side_channels,
              groups,
              pretrained=False,
              ctx=cpu(),
              **kwargs):
    if first_block_channels == 108:
        block_channels = [12, 108, 216, 432]
    elif first_block_channels == 128:
        block_channels = [12, 128, 256, 512]
    elif first_block_channels == 160:
        block_channels = [16, 160, 320, 640]
    elif first_block_channels == 228:
        block_channels = [24, 228, 456, 912]
    elif first_block_channels == 256:
        block_channels = [24, 256, 512, 1024]
    elif first_block_channels == 348:
        block_channels = [24, 348, 696, 1392]
    elif first_block_channels == 352:
        block_channels = [24, 352, 704, 1408]
    elif first_block_channels == 456:
        block_channels = [48, 456, 912, 1824]
    else:
        raise ValueError("The {} of `first_block_channels` is not supported".format(first_block_channels))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = MENet(
        block_channels=block_channels,
        side_channels=side_channels,
        groups=groups,
        **kwargs)
    return net


def menet108_8x1_g3(**kwargs):
    return get_menet(108, 8, 3, **kwargs)


def menet128_8x1_g4(**kwargs):
    return get_menet(128, 8, 4, **kwargs)


def menet160_8x1_g8(**kwargs):
    return get_menet(160, 8, 8, **kwargs)


def menet228_12x1_g3(**kwargs):
    return get_menet(228, 12, 3, **kwargs)


def menet256_12x1_g4(**kwargs):
    return get_menet(256, 12, 4, **kwargs)


def menet348_12x1_g3(**kwargs):
    return get_menet(348, 12, 3, **kwargs)


def menet352_12x1_g8(**kwargs):
    return get_menet(352, 12, 8, **kwargs)


def menet456_24x1_g3(**kwargs):
    return get_menet(456, 24, 3, **kwargs)


if __name__ == "__main__":
    import numpy as np
    import mxnet as mx
    net = menet228_12x1_g3()
    net.initialize(ctx=mx.gpu(0))
    input = mx.nd.zeros((1, 3, 224, 224), ctx=mx.gpu(0))
    output = net(input)
    #print("output={}".format(output))
    #print("net={}".format(net))

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    print("weight_count={}".format(weight_count))

