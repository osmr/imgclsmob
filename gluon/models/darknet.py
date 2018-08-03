"""
    DarkNet, implemented in Gluon.
    Original paper: 'Darknet: Open source neural networks in c'
"""

from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class DarkConv(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 **kwargs):
        super(DarkConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)
            #self.bn = nn.BatchNorm(in_channels=out_channels, momentum=0.01)
            self.activ = nn.LeakyReLU(alpha=0.1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


def dark_conv1x1(in_channels,
                 out_channels):
    return DarkConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding=0)


def dark_conv3x3(in_channels,
                 out_channels):
    return DarkConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1)


def dark_convYxY(in_channels,
                 out_channels,
                 pointwise=True):
    if pointwise:
        return dark_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)
    else:
        return dark_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels)


class DarkNet(HybridBlock):

    def __init__(self,
                 channels,
                 odd_pointwise,
                 avg_pool_size,
                 cls_activ,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(DarkNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(dark_convYxY(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            pointwise=(len(channels_per_stage) > 1) and not(((j + 1) % 2 == 1) ^ odd_pointwise)))
                        in_channels = out_channels
                    if i != len(channels) - 1:
                        stage.add(nn.MaxPool2D(
                            pool_size=2,
                            strides=2))
                self.features.add(stage)

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Conv2D(
                channels=classes,
                kernel_size=1,
                in_channels=in_channels))
            if cls_activ:
                self.output.add(nn.LeakyReLU(alpha=0.1))
            self.output.add(nn.AvgPool2D(
                pool_size=avg_pool_size,
                strides=1))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_darknet(version,
                pretrained=False,
                ctx=cpu(),
                **kwargs):
    if version == 'ref':
        channels = [[16], [32], [64], [128], [256], [512], [1024]]
        odd_pointwise = False
        avg_pool_size = 3
        cls_activ = True
    elif version == 'tiny':
        channels = [[16], [32], [16, 128, 16, 128], [32, 256, 32, 256], [64, 512, 64, 512, 128]]
        odd_pointwise = True
        avg_pool_size = 14
        cls_activ = False
    elif version == '19':
        channels = [[32], [64], [128, 64, 128], [256, 128, 256], [512, 256, 512, 256, 512], [1024, 512, 1024, 512, 1024]]
        odd_pointwise = False
        avg_pool_size = 7
        cls_activ = False
    else:
        raise ValueError("Unsupported DarkNet version {}".format(version))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return DarkNet(
        channels=channels,
        odd_pointwise=odd_pointwise,
        avg_pool_size=avg_pool_size,
        cls_activ=cls_activ,
        **kwargs)


def darknet_ref(**kwargs):
    return get_darknet('ref', **kwargs)


def darknet_tiny(**kwargs):
    return get_darknet('tiny', **kwargs)


def darknet19(**kwargs):
    return get_darknet('19', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    model = darknet_tiny
    net = model()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    assert (model != darknet_ref or weight_count == 7319416)
    assert (model != darknet_tiny or weight_count == 1042104)
    assert (model != darknet19 or weight_count == 20842376)

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

