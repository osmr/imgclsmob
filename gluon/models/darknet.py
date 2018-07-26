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
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)
            self.activ = nn.LeakyReLU(alpha=0.1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class DarkNet(HybridBlock):

    def __init__(self,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(DarkNet, self).__init__(**kwargs)
        lavels_per_stage = [1, 1, 3, 3, 5, 5]
        pool_per_stage = [1, 1, 1, 1, 1, 0]
        channels = [in_channels, 32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            k = 0
            for i in range(len(lavels_per_stage)):
                stage = nn.HybridSequential(prefix='')
                for j in range(lavels_per_stage[i]):
                    stage.add(DarkConv(
                        in_channels=channels[k],
                        out_channels=channels[k + 1],
                        kernel_size=3 if (j % 2 == 0) else 1,
                        padding=1 if (j % 2 == 0) else 0))
                    k += 1
                if pool_per_stage[i] == 1:
                    stage.add(nn.MaxPool2D(
                        pool_size=2,
                        strides=2))
                self.features.add(stage)

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Conv2D(
                channels=classes,
                kernel_size=1,
                in_channels=channels[-1]))
            self.output.add(nn.AvgPool2D(pool_size=7))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_darknet19(pretrained=False,
                  ctx=cpu(),
                  **kwargs):
    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = DarkNet(**kwargs)
    return net


def darknet19(**kwargs):
    return get_darknet19(**kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    net = get_darknet19()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    #assert (weight_count == 1597096)

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

