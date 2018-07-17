"""
    ShuffleNet, implemented in Gluon.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def depthwise_conv3x3(channels,
                      stride):
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=channels,
        bias=False)


def group_conv1x1(in_channels,
                  out_channels,
                  groups):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        groups=groups,
        bias=False)


def channel_shuffle(x,
                    groups):
    """Channel Shuffle operation from ShuffleNet [arxiv: 1707.01083]
    Arguments:
        x (Tensor): tensor to shuffle.
        groups (int): groups to be split
    """
    batch, channels, height, width = x.size()
    #assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):

    def __init__(self,
                 channels,
                 groups):
        super(ChannelShuffle, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)


class ShuffleInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ShuffleInitBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class ShuffleUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 downsample,
                 ignore_group):
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        self.compress_conv1 = group_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=(1 if ignore_group else groups))
        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=(1 if ignore_group else groups))
        self.dw_conv2 = depthwise_conv3x3(
            channels=mid_channels,
            stride=(2 if self.downsample else 1))
        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.expand_conv3 = group_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            groups=groups)
        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.activ(self.compress_bn1(self.compress_conv1(x)))
        out = self.c_shuffle(out)
        out = self.dw_bn2(self.dw_conv2(out))
        out = self.expand_bn3(self.expand_conv3(out))
        if self.downsample:
            identity = self.avgpool(identity)
            out = torch.cat((out, identity), dim=1)
        else:
            out = out + identity
        out = self.activ(out)
        return out


class ShuffleNet(nn.Module):

    def __init__(self,
                 groups,
                 stage_out_channels,
                 num_classes=1000):
        super(ShuffleNet, self).__init__()
        stage_num_blocks = [4, 8, 4]
        input_channels = 3

        self.features = nn.Sequential()
        self.features.add(ShuffleInitBlock(
            in_channels=input_channels,
            out_channels=stage_out_channels[0]))

        for i in range(len(stage_num_blocks)):
            stage = nn.Sequential()
            in_channels_i = stage_out_channels[i]
            out_channels_i = stage_out_channels[i + 1]
            for j in range(stage_num_blocks[i]):
                stage.add(ShuffleUnit(
                    in_channels=(in_channels_i if j == 0 else out_channels_i),
                    out_channels=out_channels_i,
                    groups=groups,
                    downsample=(j == 0),
                    ignore_group=(i == 0 and j == 0)))
            self.features.add(stage)

        self.features.add(nn.AvgPool2d(kernel_size=7))

        self.output = nn.Linear(
            in_features=stage_out_channels[-1],
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_shufflenet(scale,
                   groups,
                   pretrained=False,
                   **kwargs):
    if groups == 1:
        stage_out_channels = [24, 144, 288, 576]
    elif groups == 2:
        stage_out_channels = [24, 200, 400, 800]
    elif groups == 3:
        stage_out_channels = [24, 240, 480, 960]
    elif groups == 4:
        stage_out_channels = [24, 272, 544, 1088]
    elif groups == 8:
        stage_out_channels = [24, 384, 768, 1536]
    else:
        raise ValueError("The {} of groups is not supported".format(groups))
    stage_out_channels = (np.array(stage_out_channels) * scale).astype(np.int)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = ShuffleNet(
        groups=groups,
        stage_out_channels=stage_out_channels,
        **kwargs)
    return net


def shufflenet1_0_g1(**kwargs):
    return get_shufflenet(1.0, 1, **kwargs)


def shufflenet1_0_g2(**kwargs):
    return get_shufflenet(1.0, 2, **kwargs)


def shufflenet1_0_g3(**kwargs):
    return get_shufflenet(1.0, 3, **kwargs)


def shufflenet1_0_g4(**kwargs):
    return get_shufflenet(1.0, 4, **kwargs)


def shufflenet1_0_g8(**kwargs):
    return get_shufflenet(1.0, 8, **kwargs)


def shufflenet0_5_g1(**kwargs):
    return get_shufflenet(0.5, 1, **kwargs)


def shufflenet0_5_g2(**kwargs):
    return get_shufflenet(0.5, 2, **kwargs)


def shufflenet0_5_g3(**kwargs):
    return get_shufflenet(0.5, 3, **kwargs)


def shufflenet0_5_g4(**kwargs):
    return get_shufflenet(0.5, 4, **kwargs)


# def shufflenet0_5_g8(**kwargs):
#     return get_shufflenet(0.5, 8, **kwargs)


def shufflenet0_25_g1(**kwargs):
    return get_shufflenet(0.25, 1, **kwargs)


def shufflenet0_25_g2(**kwargs):
    return get_shufflenet(0.25, 2, **kwargs)


def shufflenet0_25_g3(**kwargs):
    return get_shufflenet(0.25, 3, **kwargs)


def shufflenet0_25_g4(**kwargs):
    return get_shufflenet(0.25, 4, **kwargs)


def shufflenet0_25_g8(**kwargs):
    return get_shufflenet(0.25, 8, **kwargs)
