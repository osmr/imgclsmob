"""
    ResNet, implemented in Gluon.
    Original paper: 'Deep Residual Learning for Image Recognition'
"""

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class PreResConv(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 **kwargs):
        super(PreResConv, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(in_channels=in_channels)
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
        x_pre_activ = x
        x = self.conv(x)
        return x, x_pre_activ


def conv1x1(in_channels,
            out_channels,
            strides):
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        use_bias=False,
        in_channels=in_channels)


def preres_conv1x1(in_channels,
                   out_channels,
                   strides):
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0)


def preres_conv3x3(in_channels,
                   out_channels,
                   strides):
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1)


class PreResBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(PreResBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.conv1 = preres_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides)
            self.conv2 = preres_conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1)

    def hybrid_forward(self, F, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        return x, x_pre_activ


class PreResBottleneck(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 conv1_stride,
                 **kwargs):
        super(PreResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = preres_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=(strides if conv1_stride else 1))
            self.conv2 = preres_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if conv1_stride else strides))
            self.conv3 = preres_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=1)

    def hybrid_forward(self, F, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        x, _ = self.conv3(x)
        return x, x_pre_activ


class PreResUnit(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck,
                 conv1_stride=True,
                 **kwargs):
        super(PreResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = PreResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    conv1_stride=conv1_stride)
            else:
                self.body = PreResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)
            if self.resize_identity:
                self.resize_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)

    def hybrid_forward(self, F, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.resize_conv(x_pre_activ)
        x = x + identity
        return x


class PreResInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(PreResInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=7,
                strides=2,
                padding=3,
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


class PreResNet(HybridBlock):

    def __init__(self,
                 layers,
                 channels,
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(PreResNet, self).__init__(**kwargs)
        assert (len(layers) == len(channels) - 1)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(PreResInitBlock(
                in_channels=in_channels,
                out_channels=channels[0]))
            for i, layers_per_stage in enumerate(layers):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    in_channels = channels[i]
                    out_channels = channels[i + 1]
                    for j in range(layers_per_stage):
                        strides = 1 if (i == 0) or (j != 0) else 2
                        stage.add(PreResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.BatchNorm(in_channels=channels[-1]))
            self.features.add(nn.Activation('relu'))

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


def get_preresnet(version,
                  pretrained=False,
                  ctx=cpu(),
                  **kwargs):
    if version == '18':
        layers = [2, 2, 2, 2]
        channels = [64, 64, 128, 256, 512]
        bottleneck = False
        conv1_stride = True
    else:
        raise ValueError("Unsupported ResNet version {}".format(version))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = PreResNet(
        layers=layers,
        channels=channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)
    return net


def preresnet18(**kwargs):
    return get_preresnet('18', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    net = preresnet18()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    assert (weight_count == 11687848)
    #assert (weight_count == 1042104)
    #assert (weight_count == 20842376)

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

