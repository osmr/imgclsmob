"""
    ResNet, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition'
"""

import torch.nn as nn
import torch.nn.init as init


class ResConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activate):
        super(ResConv, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def res_conv1x1(in_channels,
                out_channels,
                stride,
                activate):
    return ResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        activate=activate)


def res_conv3x3(in_channels,
                out_channels,
                stride,
                activate):
    return ResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        activate=activate)


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(ResBlock, self).__init__()
        self.conv1 = res_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            activate=True)
        self.conv2 = res_conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 conv1_stride):
        super(ResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = res_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1),
            activate=True)
        self.conv2 = res_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride),
            activate=True)
        self.conv3 = res_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 conv1_stride=True):
        super(ResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv1_stride=conv1_stride)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = res_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activate=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class ResInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResInitBlock, self).__init__()
        self.conv = ResConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            activate=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResNet(nn.Module):

    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 num_classes=1000):
        super(ResNet, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
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


def get_resnet(blocks,
               conv1_stride=True,
               width_scale=1.0,
               pretrained=False,
               **kwargs):
    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14:
        layers = [2, 2, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return ResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)


def resnet10(**kwargs):
    return get_resnet(blocks=10, **kwargs)


def resnet12(**kwargs):
    return get_resnet(blocks=12, **kwargs)


def resnet14(**kwargs):
    return get_resnet(blocks=14, **kwargs)


def resnet16(**kwargs):
    return get_resnet(blocks=16, **kwargs)


def resnet18(**kwargs):
    return get_resnet(blocks=18, **kwargs)


def resnet18_w3d4(**kwargs):
    return get_resnet(blocks=18, width_scale=0.75, **kwargs)


def resnet18_wd2(**kwargs):
    return get_resnet(blocks=18, width_scale=0.5, **kwargs)


def resnet18_wd4(**kwargs):
    return get_resnet(blocks=18, width_scale=0.25, **kwargs)


def resnet34(**kwargs):
    return get_resnet(blocks=34, **kwargs)


def resnet50(**kwargs):
    return get_resnet(blocks=50, **kwargs)


def resnet50b(**kwargs):
    return get_resnet(blocks=50, conv1_stride=False, **kwargs)


def resnet101(**kwargs):
    return get_resnet(blocks=101, **kwargs)


def resnet101b(**kwargs):
    return get_resnet(blocks=101, conv1_stride=False, **kwargs)


def resnet152(**kwargs):
    return get_resnet(blocks=152, **kwargs)


def resnet152b(**kwargs):
    return get_resnet(blocks=152, conv1_stride=False, **kwargs)


def resnet200(**kwargs):
    return get_resnet(blocks=200, **kwargs)


def resnet200b(**kwargs):
    return get_resnet(blocks=200, conv1_stride=False, **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    global TESTING
    TESTING = True

    models = [
        resnet18,
        resnet34,
        resnet50,
        resnet50b,
        resnet101,
        resnet101b,
        resnet152,
        resnet152b,
    ]

    for model in models:

        net = model()

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        assert (model != resnet18 or weight_count == 11689512)  # resnet18_v1
        assert (model != resnet34 or weight_count == 21797672)  # resnet34_v1
        assert (model != resnet50 or weight_count == 25557032)  # resnet50_v1b; resnet50_v1 -> 25575912
        assert (model != resnet50b or weight_count == 25557032)  # resnet50_v1b; resnet50_v1 -> 25575912
        assert (model != resnet101 or weight_count == 44549160)  # resnet101_v1b
        assert (model != resnet101b or weight_count == 44549160)  # resnet101_v1b
        assert (model != resnet152 or weight_count == 60192808)  # resnet152_v1b
        assert (model != resnet152b or weight_count == 60192808)  # resnet152_v1b

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

