"""
    MobileNetV2, implemented in Gluon.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks'
"""

__all__ = ['MobileNetV2', 'mobilenetv2_w1', 'mobilenetv2_w3d4', 'mobilenetv2_wd2', 'mobilenetv2_wd4']

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


TESTING = False


class ReLU6(nn.HybridBlock):

    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


class MobnetConv(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups,
                 activate,
                 **kwargs):
        super(MobnetConv, self).__init__(**kwargs)
        self.activate = activate

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
            if self.activate:
                self.activ = ReLU6()

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def mobnet_conv1x1(in_channels,
                   out_channels,
                   activate):
    return MobnetConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        groups=1,
        activate=activate)


def mobnet_dwconv3x3(in_channels,
                     out_channels,
                     strides,
                     activate):
    return MobnetConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=out_channels,
        activate=activate)


class LinearBottleneck(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 expansion,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.residual = (in_channels == out_channels) and (strides == 1)
        mid_channels = in_channels * 6 if expansion else in_channels

        with self.name_scope():
            self.conv1 = mobnet_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                activate=True)
            self.conv2 = mobnet_dwconv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                activate=True)
            self.conv3 = mobnet_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                activate=False)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV2(HybridBlock):

    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(MobnetConv(
                in_channels=in_channels,
                out_channels=init_block_channels,
                kernel_size=3,
                strides=2,
                padding=1,
                groups=1,
                activate=True))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        expansion = (i != 0) or (j != 0)
                        stage.add(LinearBottleneck(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            expansion=expansion))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(mobnet_conv1x1(
                in_channels=in_channels,
                out_channels=final_block_channels,
                activate=True))
            in_channels = final_block_channels
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Conv2D(
                channels=classes,
                kernel_size=1,
                use_bias=False,
                in_channels=in_channels))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenetv2(width_scale,
                    pretrained=False,
                    ctx=cpu(),
                    **kwargs):
    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [[]])

    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = int(final_block_channels * width_scale)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MobileNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)


def mobilenetv2_w1(**kwargs):
    return get_mobilenetv2(1, **kwargs)


def mobilenetv2_w3d4(**kwargs):
    return get_mobilenetv2(0.75, **kwargs)


def mobilenetv2_wd2(**kwargs):
    return get_mobilenetv2(0.5, **kwargs)


def mobilenetv2_wd4(**kwargs):
    return get_mobilenetv2(0.25, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    models = [
        mobilenetv2_w1,
        mobilenetv2_w3d4,
        mobilenetv2_wd2,
        mobilenetv2_wd4,
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
        assert (model != mobilenetv2_w1 or weight_count == 3504960)
        assert (model != mobilenetv2_w3d4 or weight_count == 2627592)
        assert (model != mobilenetv2_wd2 or weight_count == 1964736)
        assert (model != mobilenetv2_wd4 or weight_count == 1516392)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

