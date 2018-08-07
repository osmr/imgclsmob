"""
    PreResNet, implemented in Gluon.
    Original paper: 'Identity Mappings in Deep Residual Networks.'
"""

__all__ = ['PreResNet', 'preresnet10', 'preresnet12', 'preresnet14', 'preresnet16', 'preresnet18', 'preresnet18_w3d4',
           'preresnet18_wd2', 'preresnet18_wd4', 'preresnet34', 'preresnet50', 'preresnet50b', 'preresnet101',
           'preresnet101b', 'preresnet152', 'preresnet152b']

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
    """
    PreResNet model from 'Identity Mappings in Deep Residual Networks.'

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(PreResNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(PreResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
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
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_preresnet(blocks,
                  conv1_stride=True,
                  width_scale=1.0,
                  pretrained=False,
                  ctx=cpu(),
                  **kwargs):
    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14:
        layers = [2, 2, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
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
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return PreResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)


def preresnet10(**kwargs):
    return get_preresnet(blocks=10, **kwargs)


def preresnet12(**kwargs):
    return get_preresnet(blocks=12, **kwargs)


def preresnet14(**kwargs):
    return get_preresnet(blocks=14, **kwargs)


def preresnet16(**kwargs):
    return get_preresnet(blocks=16, **kwargs)


def preresnet18(**kwargs):
    return get_preresnet(blocks=18, **kwargs)


def preresnet18_w3d4(**kwargs):
    return get_preresnet(blocks=18, width_scale=0.75, **kwargs)


def preresnet18_wd2(**kwargs):
    return get_preresnet(blocks=18, width_scale=0.5, **kwargs)


def preresnet18_wd4(**kwargs):
    return get_preresnet(blocks=18, width_scale=0.25, **kwargs)


def preresnet34(**kwargs):
    return get_preresnet(blocks=34, **kwargs)


def preresnet50(**kwargs):
    return get_preresnet(blocks=50, **kwargs)


def preresnet50b(**kwargs):
    return get_preresnet(blocks=50, conv1_stride=False, **kwargs)


def preresnet101(**kwargs):
    return get_preresnet(blocks=101, **kwargs)


def preresnet101b(**kwargs):
    return get_preresnet(blocks=101, conv1_stride=False, **kwargs)


def preresnet152(**kwargs):
    return get_preresnet(blocks=152, **kwargs)


def preresnet152b(**kwargs):
    return get_preresnet(blocks=152, conv1_stride=False, **kwargs)


def preresnet200(**kwargs):
    return get_preresnet(blocks=200, **kwargs)


def preresnet200b(**kwargs):
    return get_preresnet(blocks=200, conv1_stride=False, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    models = [
        preresnet10,
        preresnet12,
        preresnet14,
        preresnet16,
        preresnet18,
        preresnet18_w3d4,
        preresnet18_wd2,
        preresnet18_wd4,
        preresnet34,
        preresnet50,
        preresnet50b,
        preresnet101,
        preresnet101b,
        preresnet152,
        preresnet152b,
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
        assert (model != preresnet10 or weight_count == 5417128)
        assert (model != preresnet12 or weight_count == 5491112)
        assert (model != preresnet14 or weight_count == 5786536)
        assert (model != preresnet16 or weight_count == 6967208)
        assert (model != preresnet18 or weight_count == 11687848)  # resnet18_v2
        assert (model != preresnet18_w3d4 or weight_count == 6674104)
        assert (model != preresnet18_wd2 or weight_count == 3055048)
        assert (model != preresnet18_wd4 or weight_count == 830680)
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

