from __future__ import division

__all__ = ['resnestbc14b', 'resnestbc26b', 'resnest50b', 'resnest101b', 'resnest200b', 'resnest269b']

from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon.nn import Conv2D, HybridBlock, BatchNorm, Activation
from common import conv1x1_block, conv3x3_block, SABlock


class SplitAttentionConv(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 dilation=(1, 1),
                 groups=1,
                 use_bias=False):
        super(SplitAttentionConv, self).__init__()
        radix = 2
        reduction = 2

        assert (in_channels == out_channels)

        self.radix = radix
        self.cardinality = groups
        self.out_channels = out_channels

        self.conv = Conv2D(
            out_channels * radix,
            kernel_size,
            strides,
            padding,
            dilation,
            groups=(groups * radix),
            in_channels=in_channels,
            use_bias=use_bias)
        self.bn = BatchNorm(in_channels=out_channels * radix)
        self.relu = Activation('relu')

        self.sa = SABlock(
            out_channels=out_channels,
            groups=groups,
            radix=radix)

        # inter_channels = max(in_channels * radix // 2 // reduction, 32)
        # self.att_conv1 = Conv2D(
        #     inter_channels,
        #     1,
        #     in_channels=out_channels,
        #     groups=self.cardinality)
        # self.bn1 = BatchNorm(in_channels=inter_channels)
        # self.relu1 = Activation('relu')
        #
        # self.att_conv2 = Conv2D(
        #     out_channels * radix,
        #     1,
        #     in_channels=inter_channels,
        #     groups=self.cardinality)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.sa(x)
        return x

        # x = F.reshape(x.expand_dims(1), (0, self.radix, self.out_channels, 0, 0))
        # # x = x.reshape((0, -4, self.radix, -1, -2))
        #
        # w = F.sum(x, axis=1)
        # w = F.contrib.AdaptiveAvgPooling2D(w, 1)
        #
        # w = self.att_conv1(w)
        # w = self.bn1(w)
        # w = self.relu1(w)
        #
        # w = self.att_conv2(w)
        #
        # w = w.reshape((0, self.cardinality, self.radix, -1)).swapaxes(1, 2)
        #
        # w = F.softmax(w, axis=1).reshape((0, self.radix, -1, 1, 1))
        #
        # outs = F.broadcast_mul(w, x)
        # out = F.sum(outs, axis=1)
        # return out


class Bottleneck(HybridBlock):
    expansion = 4

    def __init__(self,
                 channels,
                 strides=1,
                 downsample=None,
                 in_channels=None):
        super(Bottleneck, self).__init__()
        self.avd = (strides > 1)

        mid_channels = channels
        out_channels = channels * 4

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=1)

        self.conv2 = SplitAttentionConv(
            out_channels=mid_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            dilation=1,
            groups=1,
            use_bias=False,
            in_channels=mid_channels)

        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

        if self.avd:
            self.avd_layer = nn.AvgPool2D(3, strides, padding=1)

        self.relu3 = nn.Activation('relu')
        self.downsample = downsample

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.avd:
            out = self.avd_layer(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out


class SEInitBlock(HybridBlock):
    """
    SENet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(SEInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv3 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


class ResNeSt(HybridBlock):
    def __init__(self,
                 layers,
                 stem_width=32,
                 dropblock_prob=0.0,
                 final_drop=0.0,
                 input_size=224,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 244),
                 classes=1000):
        self.in_size = in_size
        self.classes = classes

        self.inplanes = stem_width * 2

        super(ResNeSt, self).__init__(prefix='resnest_')

        with self.name_scope():
            self.init_block = SEInitBlock(
                in_channels=in_channels,
                out_channels=(stem_width * 2),
                bn_use_global_stats=bn_use_global_stats)

            self.layer1 = self._make_layer(
                stage_index=1,
                planes=64,
                blocks=layers[0],
                strides=1)

            self.layer2 = self._make_layer(
                stage_index=2,
                planes=128,
                blocks=layers[1],
                strides=2)

            self.layer3 = self._make_layer(
                stage_index=3,
                planes=256,
                blocks=layers[2],
                strides=2)

            self.layer4 = self._make_layer(
                stage_index=4,
                planes=512,
                blocks=layers[3],
                strides=2)

            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()

            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(
                in_units=512 * Bottleneck.expansion,
                units=classes)

    def _make_layer(self,
                    stage_index,
                    planes,
                    blocks,
                    strides):
        downsample = None
        if strides != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.HybridSequential(prefix='down%d_' % stage_index)
            with downsample.name_scope():
                downsample.add(nn.AvgPool2D(
                    pool_size=strides,
                    strides=strides,
                    ceil_mode=True,
                    count_include_pad=False))
                downsample.add(nn.Conv2D(
                    channels=planes * Bottleneck.expansion,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    in_channels=self.inplanes))
                downsample.add(BatchNorm(in_channels=planes * Bottleneck.expansion))

        layers = nn.HybridSequential(prefix='layers%d_' % stage_index)
        with layers.name_scope():
            layers.add(Bottleneck(
                planes,
                strides=strides,
                downsample=downsample,
                in_channels=self.inplanes))

            self.inplanes = planes * Bottleneck.expansion
            for i in range(1, blocks):
                layers.add(Bottleneck(
                    planes,
                    in_channels=self.inplanes))

        return layers

    def hybrid_forward(self, F, x):
        x = self.init_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


def resnestbc14b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(
        layers=[1, 1, 1, 1],
        dropblock_prob=0.0,
        **kwargs)
    return model


def resnestbc26b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(
        layers=[2, 2, 2, 2],
        dropblock_prob=0.1,
        **kwargs)
    return model


def resnest50b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        dropblock_prob=0.1,
        **kwargs)
    return model


def resnest101b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(
        layers=[3, 4, 23, 3],
        stem_width=64,
        dropblock_prob=0.1,
        **kwargs)
    return model


def resnest200b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(
        layers=[3, 24, 36, 3],
        stem_width=64,
        dropblock_prob=0.1,
        final_drop=0.2,
        **kwargs)
    return model


def resnest269b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(
        layers=[3, 30, 48, 8],
        stem_width=64,
        dropblock_prob=0.1,
        final_drop=0.2,
        **kwargs)
    return model


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        resnestbc14b,
        resnestbc26b,
        resnest50b,
        resnest101b,
        resnest200b,
        resnest269b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnestbc14b or weight_count == 10611688)
        assert (model != resnestbc26b or weight_count == 17069448)
        assert (model != resnest50b or weight_count == 27483240)
        assert (model != resnest101b or weight_count == 48275016)
        assert (model != resnest200b or weight_count == 70201544)
        assert (model != resnest269b or weight_count == 110929480)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
