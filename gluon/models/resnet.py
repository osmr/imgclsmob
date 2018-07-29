"""
    ResNet, implemented in Gluon.
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
                 activate,
                 **kwargs):
        super(ResConv, self).__init__(**kwargs)
        self.activate = activate
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)
            if self.activate:
                self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def res_conv1x1(in_channels,
                out_channels,
                strides,
                activate):
    return ResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        activate=activate)


def res_conv3x3(in_channels,
                out_channels,
                strides,
                activate):
    return ResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        activate=activate)


class ResBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = res_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activate=True)
            self.conv2 = res_conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                activate=False)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 conv1_stride,
                 **kwargs):
        super(ResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = res_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=(strides if conv1_stride else 1),
                activate=True)
            self.conv2 = res_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if conv1_stride else strides),
                activate=True)
            self.conv3 = res_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=1,
                activate=False)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResUnit(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck,
                 conv1_stride=True,
                 **kwargs):
        super(ResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    conv1_stride=conv1_stride)
            else:
                self.body = ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)
            if self.resize_identity:
                self.identity_conv = res_conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    activate=False)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
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
                padding=3,
                activate=True)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResNet(HybridBlock):

    def __init__(self,
                 layers,
                 channels,
                 bottleneck,
                 conv1_stride,
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
                        stage.add(ResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride))
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
    if version == '18':
        layers = [2, 2, 2, 2]
        channels = [64, 64, 128, 256, 512]
        bottleneck = False
        conv1_stride = True
    elif version == '34':
        layers = [3, 4, 6, 3]
        channels = [64, 64, 128, 256, 512]
        bottleneck = False
        conv1_stride = True
    elif version == '50':
        layers = [3, 4, 6, 3]
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True
        conv1_stride = True
    elif version == '50b':
        layers = [3, 4, 6, 3]
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True
        conv1_stride = False
    elif version == '101':
        layers = [3, 4, 23, 3]
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True
        conv1_stride = True
    elif version == '101b':
        layers = [3, 4, 23, 3]
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True
        conv1_stride = False
    elif version == '152':
        layers = [3, 8, 36, 3]
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True
        conv1_stride = True
    elif version == '152b':
        layers = [3, 8, 36, 3]
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True
        conv1_stride = False
    else:
        raise ValueError("Unsupported ResNet version {}".format(version))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = ResNet(
        layers=layers,
        channels=channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)
    return net


def resnet18(**kwargs):
    return get_resnet('18', **kwargs)


def resnet34(**kwargs):
    return get_resnet('34', **kwargs)


def resnet50(**kwargs):
    return get_resnet('50', **kwargs)


def resnet50b(**kwargs):
    return get_resnet('50b', **kwargs)


def resnet101(**kwargs):
    return get_resnet('101', **kwargs)


def resnet101b(**kwargs):
    return get_resnet('101b', **kwargs)


def resnet152(**kwargs):
    return get_resnet('152', **kwargs)


def resnet152b(**kwargs):
    return get_resnet('152b', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    net = resnet152b()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    #assert (weight_count == 11689512)  # resnet18_v1
    #assert (weight_count == 21797672)  # resnet34_v1
    #assert (weight_count == 25557032)  # resnet50_v1b; resnet50_v1 -> 25575912
    #assert (weight_count == 44549160)  # resnet101_v1b
    assert (weight_count == 60192808)  # resnet152_v1b

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

