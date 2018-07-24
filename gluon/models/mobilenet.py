"""
    MobileNet & FD-MobileNet.
    Original papers: 
    - 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'
    - 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'
"""

import numpy as np
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class ConvBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 groups=1,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class DwsConvBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.dw_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                strides=strides,
                padding=1,
                groups=in_channels)
            self.pw_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MobileNet(HybridBlock):

    def __init__(self,
                 channels,
                 strides,
                 classes=1000,
                 **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        input_channels = 3

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(ConvBlock(
                in_channels=input_channels,
                out_channels=channels[0],
                kernel_size=3,
                strides=2,
                padding=1))
            for i in range(len(strides)):
                self.features.add(DwsConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    strides=strides[i]))
            self.features.add(nn.AvgPool2D(pool_size=7))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(
                units=classes,
                in_units=channels[-1])

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenet(scale,
                  pretrained=False,
                  ctx=cpu(),
                  **kwargs):
    channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    channels = (np.array(channels) * scale).astype(np.int)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MobileNet(channels, strides, **kwargs)


def get_fd_mobilenet(scale,
                     pretrained=False,
                     ctx=cpu(),
                     **kwargs):
    channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 1024]
    strides = [2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1]
    channels = (np.array(channels) * scale).astype(np.int)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MobileNet(channels, strides, **kwargs)


def mobilenet1_0(**kwargs):
    return get_mobilenet(1.0, **kwargs)


def mobilenet0_75(**kwargs):
    return get_mobilenet(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    return get_mobilenet(0.5, **kwargs)


def mobilenet0_25(**kwargs):
    return get_mobilenet(0.25, **kwargs)


def fd_mobilenet1_0(**kwargs):
    return get_fd_mobilenet(1.0, **kwargs)


def fd_mobilenet0_75(**kwargs):
    return get_fd_mobilenet(0.75, **kwargs)


def fd_mobilenet0_5(**kwargs):
    return get_fd_mobilenet(0.5, **kwargs)


def fd_mobilenet0_25(**kwargs):
    return get_fd_mobilenet(0.25, **kwargs)


if __name__ == "__main__":
    import numpy as np
    import mxnet as mx
    net = fd_mobilenet0_5()
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

