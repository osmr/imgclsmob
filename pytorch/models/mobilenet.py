"""
    MobileNet & FD-MobileNet.
    Original papers: 
    - 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'
    - 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'
"""

import numpy as np
import torch.nn as nn
import torch.nn.init as init


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class DwsConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(DwsConvBlock, self).__init__()

        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MobileNet(nn.Module):

    def __init__(self,
                 channels,
                 strides,
                 num_classes=1000):
        super(MobileNet, self).__init__()
        input_channels = 3

        self.features = nn.Sequential()
        self.features.add_module("init_block", ConvBlock(
            in_channels=input_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=2,
            padding=1))
        for i in range(len(strides)):
            self.features.add_module("block_{}".format(i + 1), DwsConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                stride=strides[i]))
        self.features.add_module('final_pool', nn.AvgPool2d(kernel_size=7))

        self.output = nn.Linear(
            in_features=channels[-1],
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv' in name:
                init.kaiming_normal(module.weight, mode='fan_in')
            elif name == 'conv_0' or 'pw_conv' in name:
                init.kaiming_normal(module.weight, mode='fan_out')
            elif 'bn' in name:
                init.constant(module.weight, 1)
                init.constant(module.bias, 0)
            elif 'output' in name:
                init.kaiming_normal(module.weight, mode='fan_out')
                init.constant(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_mobilenet(scale,
                  pretrained=False,
                  **kwargs):
    channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    channels = (np.array(channels) * scale).astype(np.int)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MobileNet(channels, strides, **kwargs)


def get_fd_mobilenet(scale,
                     pretrained=False,
                     **kwargs):
    channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 1024]
    strides = [2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1]
    channels = (np.array(channels) * scale).astype(np.int)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MobileNet(channels, strides, **kwargs)


def mobilenet1_0(**kwargs):
    return get_mobilenet(1.0, **kwargs)


def mobilenet0_75(**kwargs):
    return get_mobilenet(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    return get_mobilenet(0.5, **kwargs)


def mobilenet0_25(**kwargs):
    return get_mobilenet(0.25, **kwargs)


def fd_mobilenet1_0(**kwargs):
    return get_fd_mobilenet(1.0, **kwargs)


def fd_mobilenet0_75(**kwargs):
    return get_fd_mobilenet(0.75, **kwargs)


def fd_mobilenet0_5(**kwargs):
    return get_fd_mobilenet(0.5, **kwargs)


def fd_mobilenet0_25(**kwargs):
    return get_fd_mobilenet(0.25, **kwargs)
