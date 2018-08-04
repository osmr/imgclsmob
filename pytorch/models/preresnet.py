"""
    PreResNet, implemented in PyTorch.
    Original paper: 'Identity Mappings in Deep Residual Networks'
"""

import torch.nn as nn
import torch.nn.init as init


class PreResConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(PreResConv, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x_pre_activ = x
        x = self.conv(x)
        return x, x_pre_activ


def conv1x1(in_channels,
            out_channels,
            stride):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False)


def preres_conv1x1(in_channels,
                   out_channels,
                   stride):
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0)


def preres_conv3x3(in_channels,
                   out_channels,
                   stride):
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1)


class PreResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PreResBlock, self).__init__()
        self.conv1 = preres_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.conv2 = preres_conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        return x, x_pre_activ


class PreResBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 conv1_stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = preres_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1))
        self.conv2 = preres_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride))
        self.conv3 = preres_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        x, _ = self.conv3(x)
        return x, x_pre_activ


class PreResUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 conv1_stride=True):
        super(PreResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = PreResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv1_stride=conv1_stride)
        else:
            self.body = PreResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class PreResInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(PreResInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
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


class PreResActivation(nn.Module):

    def __init__(self,
                 in_channels):
        super(PreResActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class PreResNet(nn.Module):

    def __init__(self,
                 layers,
                 channels,
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 num_classes=1000):
        super(PreResNet, self).__init__()
        assert (len(layers) == len(channels) - 1)

        self.features = nn.Sequential()
        self.features.add_module("init_block", PreResInitBlock(
            in_channels=in_channels,
            out_channels=channels[0]))
        in_channels = channels[0]
        for i, layers_per_stage in enumerate(layers):
            stage = nn.Sequential()
            out_channels = channels[i + 1]
            for j in range(layers_per_stage):
                stride = 1 if (i == 0) or (j != 0) else 2
                stage.add_module("unit{}".format(j + 1), PreResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('post_activ', PreResActivation(in_channels=channels[-1]))
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


def get_preresnet(version,
                  pretrained=False,
                  **kwargs):
    if version.endswith("b"):
        conv1_stride = False
        pure_version = version[:-1]
    else:
        conv1_stride = True
        pure_version = version

    if not pure_version.isdigit():
        raise ValueError("Unsupported PreResNet version {}".format(version))

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
        raise ValueError("Unsupported PreResNet version {}".format(version))

    if blocks < 50:
        channels = [64, 64, 128, 256, 512]
        bottleneck = False
    else:
        channels = [64, 256, 512, 1024, 2048]
        bottleneck = True

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return PreResNet(
        layers=layers,
        channels=channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)


def preresnet18(**kwargs):
    return get_preresnet('18', **kwargs)


def preresnet34(**kwargs):
    return get_preresnet('34', **kwargs)


def preresnet50(**kwargs):
    return get_preresnet('50', **kwargs)


def preresnet50b(**kwargs):
    return get_preresnet('50b', **kwargs)


def preresnet101(**kwargs):
    return get_preresnet('101', **kwargs)


def preresnet101b(**kwargs):
    return get_preresnet('101b', **kwargs)


def preresnet152(**kwargs):
    return get_preresnet('152', **kwargs)


def preresnet152b(**kwargs):
    return get_preresnet('152b', **kwargs)


def preresnet200(**kwargs):
    return get_preresnet('200', **kwargs)


def preresnet200b(**kwargs):
    return get_preresnet('200b', **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    global TESTING
    TESTING = True

    models = [
        preresnet18,
        preresnet34,
        preresnet50,
        preresnet50b,
        preresnet101,
        preresnet101b,
        preresnet152,
        preresnet152b,
    ]

    for model in models:

        net = model()

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        assert (model != preresnet18 or weight_count == 11687848)  # resnet18_v2
        assert (model != preresnet34 or weight_count == 21796008)  # resnet34_v2
        assert (model != preresnet50 or weight_count == 25549480)  # resnet50_v2
        assert (model != preresnet50b or weight_count == 25549480)  # resnet50_v2
        assert (model != preresnet101 or weight_count == 44541608)  # resnet101_v2
        assert (model != preresnet101b or weight_count == 44541608)  # resnet101_v2
        assert (model != preresnet152 or weight_count == 60185256)  # resnet152_v2
        assert (model != preresnet152b or weight_count == 60185256)  # resnet152_v2

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

