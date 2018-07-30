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
                 layers,
                 channels,
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 num_classes=1000):
        super(ResNet, self).__init__()
        assert (len(layers) == len(channels) - 1)

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=channels[0]))
        for i, layers_per_stage in enumerate(layers):
            stage = nn.Sequential()
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(layers_per_stage):
                stride = 1 if (i == 0) or (j != 0) else 2
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
            in_features=channels[-1],
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


def get_resnet(version,
               pretrained=False,
               **kwargs):
    if version.endswith("b"):
        conv1_stride = False
        pure_version = version[:-1]
    else:
        conv1_stride = True
        pure_version = version

    if not pure_version.isdigit():
        raise ValueError("Unsupported ResNet version {}".format(version))

    blocks = int(pure_version)
    if blocks == 18:
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
        raise ValueError("Unsupported ResNet version {}".format(version))

    if blocks < 50:
        channels = [64, 64, 128, 256, 512]
        bottleneck = False
    else:
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = ResNet(
        layers=layers,
        channels=channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)
    return net


def resnet18(**kwargs):
    return get_resnet('18', **kwargs)


def resnet34(**kwargs):
    return get_resnet('34', **kwargs)


def resnet50(**kwargs):
    return get_resnet('50', **kwargs)


def resnet50b(**kwargs):
    return get_resnet('50b', **kwargs)


def resnet101(**kwargs):
    return get_resnet('101', **kwargs)


def resnet101b(**kwargs):
    return get_resnet('101b', **kwargs)


def resnet152(**kwargs):
    return get_resnet('152', **kwargs)


def resnet152b(**kwargs):
    return get_resnet('152b', **kwargs)


def resnet200(**kwargs):
    return get_resnet('200', **kwargs)


def resnet200b(**kwargs):
    return get_resnet('200b', **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    global TESTING
    TESTING = True

    net = resnet34()

    net.train()
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    #assert (weight_count == 11689512)  # resnet18_v1
    assert (weight_count == 21797672)  # resnet34_v1
    #assert (weight_count == 25557032)  # resnet50_v1b; resnet50_v1 -> 25575912
    #assert (weight_count == 44549160)  # resnet101_v1b
    #assert (weight_count == 60192808)  # resnet152_v1b

    x = Variable(torch.randn(1, 3, 224, 224))
    y = net(x)
    assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

