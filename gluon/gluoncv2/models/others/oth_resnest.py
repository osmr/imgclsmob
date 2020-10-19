from __future__ import division
import math
import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon.nn import Conv2D, HybridBlock, BatchNorm, Activation

__all__ = ['oth_resnest14', 'oth_resnest26', 'oth_resnest50', 'oth_resnest101', 'oth_resnest200', 'oth_resnest269']


class DropBlock(HybridBlock):
    def __init__(self, drop_prob, block_size, c, h, w):
        super(DropBlock, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.c, self.h, self.w = c, h, w
        self.numel = c * h * w
        pad_h = max((block_size - 1), 0)
        pad_w = max((block_size - 1), 0)
        self.padding = (pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2)
        self.dtype = 'float32'

    def hybrid_forward(self, F, x):
        if not mx.autograd.is_training() or self.drop_prob <= 0:
            return x
        gamma = self.drop_prob * (self.h * self.w) / (self.block_size ** 2) / \
            ((self.w - self.block_size + 1) * (self.h - self.block_size + 1))
        # generate mask
        mask = F.random.uniform(0, 1, shape=(1, self.c, self.h, self.w), dtype=self.dtype) < gamma
        mask = F.Pooling(mask, pool_type='max',
                         kernel=(self.block_size, self.block_size), pad=self.padding)
        mask = 1 - mask
        y = F.broadcast_mul(F.broadcast_mul(x, mask),
                            (1.0 * self.numel / mask.sum(axis=0, exclude=True).expand_dims(1).expand_dims(1).expand_dims(1)))
        return y

    def cast(self, dtype):
        super(DropBlock, self).cast(dtype)
        self.dtype = dtype

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
            'drop_prob: {}, block_size{}'.format(self.drop_prob, self.block_size) +')'
        return reprstr


class SplitAttentionConv(HybridBlock):
    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, radix=2, in_channels=None, r=2,
                 norm_layer=BatchNorm, norm_kwargs=None, drop_ratio=0,
                 *args, **kwargs):
        super(SplitAttentionConv, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        inter_channels = max(in_channels*radix//2//r, 32)
        self.radix = radix
        self.cardinality = groups
        self.conv = Conv2D(channels*radix, kernel_size, strides, padding, dilation,
                           groups=groups*radix, *args, in_channels=in_channels, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn = norm_layer(in_channels=channels*radix, **norm_kwargs)
        self.relu = Activation('relu')
        self.fc1 = Conv2D(inter_channels, 1, in_channels=channels, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(in_channels=inter_channels, **norm_kwargs)
        self.relu1 = Activation('relu')
        if drop_ratio > 0:
            self.drop = nn.Dropout(drop_ratio)
        else:
            self.drop = None
        self.fc2 = Conv2D(channels*radix, 1, in_channels=inter_channels, groups=self.cardinality)
        self.channels = channels

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        if self.radix > 1:
            splited = F.reshape(x.expand_dims(1), (0, self.radix, self.channels, 0, 0))
            gap = F.sum(splited, axis=1)
        else:
            gap = x
        gap = F.contrib.AdaptiveAvgPooling2D(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        atten = self.relu1(gap)
        if self.drop:
            atten = self.drop(atten)
        atten = self.fc2(atten).reshape((0, self.cardinality, self.radix, -1)).swapaxes(1, 2)
        if self.radix > 1:
            atten = F.softmax(atten, axis=1).reshape((0, self.radix, -1, 1, 1))
        else:
            atten = F.sigmoid(atten).reshape((0, -1, 1, 1))
        if self.radix > 1:
            outs = F.broadcast_mul(atten, splited)
            out = F.sum(outs, axis=1)
        else:
            out = F.broadcast_mul(atten, x)
        return out


def _update_input_size(input_size, stride):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ih, iw = (input_size, input_size) if isinstance(input_size, int) else input_size
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    input_size = (oh, ow)
    return input_size


class Bottleneck(HybridBlock):
    """ResNeSt Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, channels, cardinality=1, bottleneck_width=64, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False,
                 dropblock_prob=0, input_size=None, use_splat=False,
                 radix=2, avd=False, avd_first=False, in_channels=None,
                 split_drop_ratio=0, **kwargs):
        super(Bottleneck, self).__init__()
        group_width = int(channels * (bottleneck_width / 64.)) * cardinality
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.dropblock_prob = dropblock_prob
        self.use_splat = use_splat
        self.avd = avd and (strides > 1 or previous_dilation != dilation)
        self.avd_first = avd_first
        if self.dropblock_prob > 0:
            self.dropblock1 = DropBlock(dropblock_prob, 3, group_width, *input_size)
            if self.avd:
                if avd_first:
                    input_size = _update_input_size(input_size, strides)
                self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
                if not avd_first:
                    input_size = _update_input_size(input_size, strides)
            else:
                input_size = _update_input_size(input_size, strides)
                self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
            self.dropblock3 = DropBlock(dropblock_prob, 3, channels * 4, *input_size)
        self.conv1 = nn.Conv2D(channels=group_width, kernel_size=1,
                               use_bias=False, in_channels=in_channels)
        self.bn1 = norm_layer(in_channels=group_width, **norm_kwargs)
        self.relu1 = nn.Activation('relu')
        if self.use_splat:
            self.conv2 = SplitAttentionConv(channels=group_width, kernel_size=3,
                                            strides=1 if self.avd else strides,
                                            padding=dilation, dilation=dilation, groups=cardinality,
                                            use_bias=False, in_channels=group_width,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                            radix=radix, drop_ratio=split_drop_ratio,
                                            **kwargs)
        else:
            self.conv2 = nn.Conv2D(channels=group_width, kernel_size=3,
                                   strides=1 if self.avd else strides,
                                   padding=dilation, dilation=dilation, groups=cardinality,
                                   use_bias=False, in_channels=group_width, **kwargs)
            self.bn2 = norm_layer(in_channels=group_width, **norm_kwargs)
            self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=channels * 4, kernel_size=1, use_bias=False,
                               in_channels=group_width)
        if not last_gamma:
            self.bn3 = norm_layer(in_channels=channels * 4, **norm_kwargs)
        else:
            self.bn3 = norm_layer(in_channels=channels * 4, gamma_initializer='zeros',
                                  **norm_kwargs)
        if self.avd:
            self.avd_layer = nn.AvgPool2D(3, strides, padding=1)
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0:
            out = self.dropblock1(out)
        out = self.relu1(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        if self.use_splat:
            out = self.conv2(out)
            if self.dropblock_prob > 0:
                out = self.dropblock2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            if self.dropblock_prob > 0:
                out = self.dropblock2(out)
            out = self.relu2(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.dropblock_prob > 0:
            out = self.dropblock3(out)

        out = out + residual
        out = self.relu3(out)

        return out


class ResNeSt(HybridBlock):
    """ ResNeSt Model
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    """
    def __init__(self,
                 layers,
                 stem_width=32,
                 dropblock_prob=0.0,
                 final_drop=0.0,
                 input_size=224,
                 in_channels=3,
                 in_size=(224, 244),
                 classes=1000):
        self.in_size = in_size
        self.classes = classes

        block = Bottleneck
        avg_down = True
        cardinality = 1
        avd = True
        avd_first = False
        use_splat = True
        bottleneck_width = 64
        radix = 2
        split_drop_ratio = 0

        dilated = False
        dilation = 1
        norm_layer = BatchNorm
        norm_kwargs = None
        last_gamma = False

        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2
        self.radix = radix
        self.split_drop_ratio = split_drop_ratio
        self.avd_first = avd_first
        super(ResNeSt, self).__init__(prefix='resnest_')
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.norm_kwargs = norm_kwargs
        with self.name_scope():
            self.conv1 = nn.HybridSequential(prefix='conv1')
            self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                     padding=1, use_bias=False, in_channels=3))
            self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
            self.conv1.add(nn.Activation('relu'))
            self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                     padding=1, use_bias=False, in_channels=stem_width))
            self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
            self.conv1.add(nn.Activation('relu'))
            self.conv1.add(nn.Conv2D(channels=stem_width * 2, kernel_size=3, strides=1,
                                     padding=1, use_bias=False, in_channels=stem_width))

            input_size = _update_input_size(input_size, 2)
            self.bn1 = norm_layer(in_channels=stem_width * 2, **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            input_size = _update_input_size(input_size, 2)
            self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma,
                                           use_splat=use_splat, avd=avd)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma,
                                           use_splat=use_splat, avd=avd)
            input_size = _update_input_size(input_size, 2)
            if dilated or dilation == 4:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4,
                                               pre_dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            elif dilation == 3:
                # special
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2, dilation=2,
                                               pre_dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            elif dilation == 2:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            else:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                input_size = _update_input_size(input_size, 2)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                input_size = _update_input_size(input_size, 2)
            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1, pre_dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False, dropblock_prob=0,
                    input_size=224, use_splat=False, avd=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_' % stage_index)
            with downsample.name_scope():
                if avg_down:
                    if pre_dilation == 1:
                        downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
                                                    ceil_mode=True, count_include_pad=False))
                    elif strides == 1:
                        downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
                                                    ceil_mode=True, count_include_pad=False))
                    else:
                        downsample.add(
                            nn.AvgPool2D(pool_size=pre_dilation * strides, strides=strides,
                                         padding=1, ceil_mode=True, count_include_pad=False))
                    downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                             strides=1, use_bias=False, in_channels=self.inplanes))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))
                else:
                    downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                             kernel_size=1, strides=strides, use_bias=False,
                                             in_channels=self.inplanes))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))

        layers = nn.HybridSequential(prefix='layers%d_' % stage_index)
        with layers.name_scope():
            if dilation in (1, 2):
                layers.add(block(planes, cardinality=self.cardinality,
                                 bottleneck_width=self.bottleneck_width,
                                 strides=strides, dilation=pre_dilation,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                 input_size=input_size, use_splat=use_splat, avd=avd,
                                 avd_first=self.avd_first, radix=self.radix,
                                 in_channels=self.inplanes, split_drop_ratio=self.split_drop_ratio))
            elif dilation == 4:
                layers.add(block(planes, cardinality=self.cardinality,
                                 bottleneck_width=self.bottleneck_width,
                                 strides=strides, dilation=pre_dilation,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                 input_size=input_size, use_splat=use_splat, avd=avd,
                                 avd_first=self.avd_first, radix=self.radix,
                                 in_channels=self.inplanes, split_drop_ratio=self.split_drop_ratio))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            input_size = _update_input_size(input_size, strides)
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(planes, cardinality=self.cardinality,
                                 bottleneck_width=self.bottleneck_width, dilation=dilation,
                                 previous_dilation=dilation, norm_layer=norm_layer,
                                 norm_kwargs=self.norm_kwargs, last_gamma=last_gamma,
                                 dropblock_prob=dropblock_prob, input_size=input_size,
                                 use_splat=use_splat, avd=avd, avd_first=self.avd_first,
                                 radix=self.radix, in_channels=self.inplanes,
                                 split_drop_ratio=self.split_drop_ratio))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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


def oth_resnest14(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(layers=[1, 1, 1, 1],
                    dropblock_prob=0.0,
                    **kwargs)
    return model


def oth_resnest26(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(layers=[2, 2, 2, 2],
                    dropblock_prob=0.1,
                    **kwargs)
    return model


def oth_resnest50(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(layers=[3, 4, 6, 3],
                    dropblock_prob=0.1,
                    **kwargs)
    return model


def oth_resnest101(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(layers=[3, 4, 23, 3],
                    stem_width=64,
                    dropblock_prob=0.1,
                    **kwargs)
    return model


def oth_resnest200(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(layers=[3, 24, 36, 3],
                    stem_width=64,
                    dropblock_prob=0.1,
                    final_drop=0.2,
                    **kwargs)
    return model


def oth_resnest269(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNeSt(layers=[3, 30, 48, 8],
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
        # oth_resnest14,
        # oth_resnest26,
        # oth_resnest50,
        oth_resnest101,
        oth_resnest200,
        oth_resnest269,
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
        assert (model != oth_resnest14 or weight_count == 10611688)
        assert (model != oth_resnest26 or weight_count == 17069448)
        assert (model != oth_resnest50 or weight_count == 27483240)
        assert (model != oth_resnest101 or weight_count == 48275016)
        assert (model != oth_resnest200 or weight_count == 70201544)
        assert (model != oth_resnest269 or weight_count == 110929480)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
