"""
    ShuffleNet, implemented in Gluon.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'
"""

from mxnet.gluon import HybridBlock


class ShuffleNet(HybridBlock):

    def __init__(self,
                 scale,
                 groups,
                 **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x


def get_shufflenet(scale,
                   groups,
                   **kwargs):
    net = ShuffleNet(scale, groups, **kwargs)
    return net


def shufflenet1_0_g1(**kwargs):
    return get_shufflenet(1.0, 1, **kwargs)


def shufflenet1_0_g2(**kwargs):
    return get_shufflenet(1.0, 2, **kwargs)

