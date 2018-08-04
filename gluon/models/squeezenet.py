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
                 channels,
                 residuals,
                 init_block_kernel_size,
                 init_block_channels,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(SqueezeInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                kernel_size=init_block_kernel_size))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    stage.add(nn.MaxPool2D(
                        pool_size=3,
                        strides=2,
                        ceil_mode=True))
                    for j, out_channels in enumerate(channels_per_stage):
                        expand_channels = out_channels // 2
                        squeeze_channels = out_channels // 8
                        stage.add(FireUnit(
                            in_channels=in_channels,
                            squeeze_channels=squeeze_channels,
                            expand1x1_channels=expand_channels,
                            expand3x3_channels=expand_channels,
                            residual=((residuals is not None) and (residuals[i][j] == 1))))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.Dropout(rate=0.5))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Conv2D(
                channels=classes,
                kernel_size=1,
                in_channels=in_channels))
            self.output.add(nn.Activation('relu'))
            self.output.add(nn.AvgPool2D(
                pool_size=13,
                strides=1))
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
        channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
        residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]
        init_block_kernel_size = 7
        init_block_channels = 96
    elif version == '1.1':
        channels = [[128, 128], [256, 256], [384, 384, 512, 512]]
        residuals = [[0, 1], [0, 1], [0, 1, 0, 1]]
        init_block_kernel_size = 3
        init_block_channels = 64
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    if not residual:
        residuals = None

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return SqueezeNet(
        channels=channels,
        residuals=residuals,
        init_block_kernel_size=init_block_kernel_size,
        init_block_channels=init_block_channels,
        **kwargs)


def squeezenet_v1_0(**kwargs):
    return get_squeezenet(version='1.0', residual=False, **kwargs)


def squeezenet_v1_1(**kwargs):
    return get_squeezenet(version='1.1', residual=False, **kwargs)


def squeezeresnet_v1_0(**kwargs):
    return get_squeezenet(version='1.0', residual=True, **kwargs)


def squeezeresnet_v1_1(**kwargs):
    return get_squeezenet(version='1.1', residual=True, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    models = [
        squeezenet_v1_0,
        squeezenet_v1_1,
        squeezeresnet_v1_0,
        squeezeresnet_v1_1,
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
        assert (model != squeezenet_v1_0 or weight_count == 1248424)
        assert (model != squeezenet_v1_1 or weight_count == 1235496)
        assert (model != squeezeresnet_v1_0 or weight_count == 1248424)
        assert (model != squeezeresnet_v1_1 or weight_count == 1235496)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

