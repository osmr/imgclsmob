"""
    PreResNet, implemented in Gluon.
    Original paper: 'Identity Mappings in Deep Residual Networks'
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
                 bn_use_global_stats,
                 **kwargs):
        super(PreResConv, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
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
                   strides,
                   bn_use_global_stats):
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        bn_use_global_stats=bn_use_global_stats)


def preres_conv3x3(in_channels,
                   out_channels,
                   strides,
                   bn_use_global_stats):
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        bn_use_global_stats=bn_use_global_stats)


class PreResBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 **kwargs):
        super(PreResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = preres_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = preres_conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        return x, x_pre_activ


class PreResBottleneck(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 conv1_stride,
                 **kwargs):
        super(PreResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = preres_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=(strides if conv1_stride else 1),
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = preres_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if conv1_stride else strides),
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = preres_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=1,
                bn_use_global_stats=bn_use_global_stats)

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
                 bn_use_global_stats,
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
                    bn_use_global_stats=bn_use_global_stats,
                    conv1_stride=conv1_stride)
            else:
                self.body = PreResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)

    def hybrid_forward(self, F, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class PreResInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
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
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
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


class PreResActivation(HybridBlock):

    def __init__(self,
                 in_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(PreResActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class PreResNet(HybridBlock):

    def __init__(self,
                 layers,
                 channels,
                 bottleneck,
                 conv1_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(PreResNet, self).__init__(**kwargs)
        assert (len(layers) == len(channels) - 1)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(PreResInitBlock(
                in_channels=in_channels,
                out_channels=channels[0],
                bn_use_global_stats=bn_use_global_stats))
            in_channels = channels[0]
            for i, layers_per_stage in enumerate(layers):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    out_channels = channels[i + 1]
                    for j in range(layers_per_stage):
                        strides = 1 if (i == 0) or (j != 0) else 2
                        stage.add(PreResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PreResActivation(
                in_channels=channels[-1],
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
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
    if version.endswith("b"):
        conv1_stride = False
        pure_version = version[:-1]
    else:
        conv1_stride = True
        pure_version = version

    if not pure_version.isdigit():
        raise ValueError("Unsupported PreResNet version {}".format(version))

    blocks = int(pure_version)
    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported PreResNet version {}".format(version))

    if blocks < 50:
        channels = [64, 64, 128, 256, 512]
        bottleneck = False
    else:
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return PreResNet(
        layers=layers,
        channels=channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)


def preresnet18(**kwargs):
    return get_preresnet('18', **kwargs)


def preresnet34(**kwargs):
    return get_preresnet('34', **kwargs)


def preresnet50(**kwargs):
    return get_preresnet('50', **kwargs)


def preresnet50b(**kwargs):
    return get_preresnet('50b', **kwargs)


def preresnet101(**kwargs):
    return get_preresnet('101', **kwargs)


def preresnet101b(**kwargs):
    return get_preresnet('101b', **kwargs)


def preresnet152(**kwargs):
    return get_preresnet('152', **kwargs)


def preresnet152b(**kwargs):
    return get_preresnet('152b', **kwargs)


def preresnet200(**kwargs):
    return get_preresnet('200', **kwargs)


def preresnet200b(**kwargs):
    return get_preresnet('200b', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    model = preresnet152b
    net = model()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    assert (model != preresnet18 or weight_count == 11687848)  # resnet18_v2
    assert (model != preresnet34 or weight_count == 21796008)  # resnet34_v2
    assert (model != preresnet50 or weight_count == 25549480)  # resnet50_v2
    assert (model != preresnet50b or weight_count == 25549480)  # resnet50_v2
    assert (model != preresnet101 or weight_count == 44541608)  # resnet101_v2
    assert (model != preresnet101b or weight_count == 44541608)  # resnet101_v2
    assert (model != preresnet152 or weight_count == 60185256)  # resnet152_v2
    assert (model != preresnet152b or weight_count == 60185256)  # resnet152_v2

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

