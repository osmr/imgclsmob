"""
    SqueezeNet, implemented in Gluon.
    Original paper: 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'
"""

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


TESTING = False


class FireConv(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 **kwargs):
        super(FireConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                in_channels=in_channels)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class FireUnit(HybridBlock):

    def __init__(self,
                 in_channels,
                 squeeze_channels,
                 expand1x1_channels,
                 expand3x3_channels,
                 residual,
                 **kwargs):
        super(FireUnit, self).__init__(**kwargs)
        self.residual = residual
        with self.name_scope():
            self.squeeze = FireConv(
                in_channels=in_channels,
                out_channels=squeeze_channels,
                kernel_size=1,
                padding=0)
            self.expand1x1 = FireConv(
                in_channels=squeeze_channels,
                out_channels=expand1x1_channels,
                kernel_size=1,
                padding=0)
            self.expand3x3 = FireConv(
                in_channels=squeeze_channels,
                out_channels=expand3x3_channels,
                kernel_size=3,
                padding=1)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = F.concat(y1, y2, dim=1)
        if self.residual:
            out = out + identity
        return out


def squeeze_pool():
    return nn.MaxPool2D(
        pool_size=3,
        strides=2,
        ceil_mode=True)


class SqueezeInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 **kwargs):
        super(SqueezeInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=2,
                in_channels=in_channels)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class SqueezeNet(HybridBlock):

    def __init__(self,
                 first_out_channels,
                 first_kernel_size,
                 pool_stages,
                 residual_stages,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)
        stage_squeeze_channels = [16, 32, 48, 64]
        stage_expand_channels = [64, 128, 192, 256]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(SqueezeInitBlock(
                in_channels=in_channels,
                out_channels=first_out_channels,
                kernel_size=first_kernel_size))
            k = 0
            pool_ind = 0
            res_ind = 0
            for i in range(len(stage_squeeze_channels)):
                for j in range(2):
                    if (pool_ind < len(pool_stages)) and (k == pool_stages[pool_ind]):
                        self.features.add(squeeze_pool())
                        pool_ind += 1
                    if (res_ind < len(residual_stages)) and (k == residual_stages[res_ind]):
                        residual = True
                        res_ind += 1
                    else:
                        residual = False
                    in_channels_ij = first_out_channels if (i == 0 and j == 0) else \
                        (2 * stage_expand_channels[i - 1] if j == 0 else 2 * stage_expand_channels[i])
                    self.features.add(FireUnit(
                        in_channels=in_channels_ij,
                        squeeze_channels=stage_squeeze_channels[i],
                        expand1x1_channels=stage_expand_channels[i],
                        expand3x3_channels=stage_expand_channels[i],
                        residual=residual))
                    k += 1
            self.features.add(nn.Dropout(rate=0.5))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Conv2D(
                channels=classes,
                kernel_size=1,
                in_channels=(2 * stage_expand_channels[-1])))
            self.output.add(nn.Activation('relu'))
            self.output.add(nn.AvgPool2D(pool_size=13))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        assert ((not TESTING) or x.shape == (1, 3, 224, 224))

        x = self.features(x)
        assert ((not TESTING) or x.shape == (1, 512, 13, 13))
        x = self.output(x)
        assert ((not TESTING) or x.shape == (1, 1000))

        return x


def get_squeezenet(version,
                   residual=False,
                   pretrained=False,
                   ctx=cpu(),
                   **kwargs):
    if version == '1.0':
        first_out_channels = 96
        first_kernel_size = 7
        pool_stages = [0, 3, 7]
    elif version == '1.1':
        first_out_channels = 64
        first_kernel_size = 3
        pool_stages = [0, 2, 4]
    else:
        raise ValueError("Unsupported SqueezeNet version {}: 1.0 or 1.1 expected".format(version))

    if residual:
        residual_stages = [1, 3, 5, 7]
    else:
        residual_stages = []

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return SqueezeNet(
        first_out_channels=first_out_channels,
        first_kernel_size=first_kernel_size,
        pool_stages=pool_stages,
        residual_stages=residual_stages,
        **kwargs)


def squeezenet1_0(**kwargs):
    return get_squeezenet(version='1.0', residual=False, **kwargs)


def squeezenet1_1(**kwargs):
    return get_squeezenet(version='1.1', residual=False, **kwargs)


def squeezeresnet1_0(**kwargs):
    return get_squeezenet(version='1.0', residual=True, **kwargs)


def squeezeresnet1_1(**kwargs):
    return get_squeezenet(version='1.1', residual=True, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    net = squeezenet1_0()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    assert (weight_count == 1248424)

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

