"""
    MobileNet, implemented in Gluon.
    Original paper: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'
"""

import numpy as np
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class MobileNet(HybridBlock):

    def __init__(self,
                 scale=1.0,
                 classes=1000,
                 **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        input_channels = 3
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        channels = (np.array(channels) * scale).astype(np.int)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(self._conv_block(
                in_channels=input_channels,
                out_channels=channels[0],
                kernel_size=3,
                strides=2,
                padding=1))
            for i in range(len(strides)):
                self.features.add(self._dws_conv_block(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    strides=strides[i]))
            self.features.add(nn.AvgPool2D(pool_size=7))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(units=classes)

    @staticmethod
    def _dws_conv_block(in_channels,
                        out_channels,
                        strides):
        dws_conv_blk = nn.HybridSequential(prefix='')
        dws_conv_blk.add(MobileNet._conv_block(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            strides=strides,
            padding=1,
            groups=in_channels))
        dws_conv_blk.add(MobileNet._conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1))
        return dws_conv_blk

    @staticmethod
    def _conv_block(in_channels,
                    out_channels,
                    kernel_size,
                    strides=1,
                    padding=0,
                    groups=1):
        conv_blk = nn.HybridSequential(prefix='')
        conv_blk.add(nn.Conv2D(
            channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=False,
            in_channels=in_channels))
        conv_blk.add(nn.BatchNorm(scale=True))
        conv_blk.add(nn.Activation('relu'))
        return conv_blk

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenet(scale,
                  pretrained=False,
                  ctx=cpu(),
                  **kwargs):
    return MobileNet(scale, **kwargs)


def mobilenet1_0(**kwargs):
    return get_mobilenet(1.0, **kwargs)


def mobilenet0_75(**kwargs):
    return get_mobilenet(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    return get_mobilenet(0.5, **kwargs)


def mobilenet0_25(**kwargs):
    return get_mobilenet(0.25, **kwargs)
