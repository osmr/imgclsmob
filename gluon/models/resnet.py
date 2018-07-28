"""
    ReskNet, implemented in Gluon.
    Original paper: 'Deep Residual Learning for Image Recognition'
"""

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class ResConv(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 **kwargs):
        super(ResConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def res_conv1x1(in_channels,
                out_channels,
                strides):
    return ResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0)


def res_conv3x3(in_channels,
                out_channels,
                strides):
    return ResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1)


class ResSimpleBlockV1(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(ResSimpleBlockV1, self).__init__(**kwargs)

        with self.name_scope():
            self.conv1 = res_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides)
            self.activ = nn.Activation('relu')
            self.conv2 = res_conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        return x


class ResBottleneckV1(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(ResBottleneckV1, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = res_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=strides)
            self.conv2 = res_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=1)
            self.conv3 = res_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=1)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.conv3(x)
        return x


class ResUnitV1(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck,
                 **kwargs):
        super(ResUnitV1, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ResBottleneckV1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)
            else:
                self.body = ResSimpleBlockV1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)
            if self.resize_identity:
                self.resize_conv = res_conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.resize_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class ResInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(ResInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = ResConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                strides=2,
                padding=3)
            self.activ = nn.Activation('relu')
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class ResNet(HybridBlock):

    def __init__(self,
                 block,
                 layers,
                 channels,
                 bottleneck,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(ResNet, self).__init__(**kwargs)
        assert (len(layers) == len(channels) - 1)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=channels[0]))
            for i, layers_per_stage in enumerate(layers):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    in_channels = channels[i]
                    out_channels = channels[i + 1]
                    for j in range(layers_per_stage):
                        strides = 1 if (i == 0) or (j != 0) else 2
                        stage.add(block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bottleneck=bottleneck))
                        in_channels = out_channels
                self.features.add(stage)

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.AvgPool2D(pool_size=7))
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=channels[-1]))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_resnet(version,
               pretrained=False,
               ctx=cpu(),
               **kwargs):
    if version == '18_v1':
        block_class = ResUnitV1
        layers = [2, 2, 2, 2]
        channels = [64, 64, 128, 256, 512]
        bottleneck = False
    else:
        raise ValueError("Unsupported ResNet version {}".format(version))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = ResNet(
        block=block_class,
        layers=layers,
        channels=channels,
        bottleneck=bottleneck,
        **kwargs)
    return net


def resnet18_v1(**kwargs):
    return get_resnet('18_v1', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    net = resnet18_v1()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    assert (weight_count == 11689512)
    #assert (weight_count == 1042104)
    #assert (weight_count == 20842376)

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

