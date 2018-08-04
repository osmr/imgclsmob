"""
    MobileNet & FD-MobileNet.
    Original papers: 
    - 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'
    - 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'
"""

import torch.nn as nn
import torch.nn.init as init


TESTING = False


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
                 first_stage_stride,
                 in_channels=3,
                 num_classes=1000):
        super(MobileNet, self).__init__()

        self.features = nn.Sequential()
        init_block_channels = channels[0][0]
        self.features.add_module("init_block", ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=2,
            padding=1))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                stage.add_module("unit{}".format(j + 1), DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
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
            if 'dw_conv.conv' in name:
                init.kaiming_normal_(module.weight, mode='fan_in')
            elif name == 'init_block.conv' or 'pw_conv.conv' in name:
                init.kaiming_normal_(module.weight, mode='fan_out')
            elif 'bn' in name:
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif 'output' in name:
                init.kaiming_normal_(module.weight, mode='fan_out')
                init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_mobilenet(version,
                  width_scale,
                  pretrained=False,
                  **kwargs):
    if version == 'orig':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
        first_stage_stride = False
    elif version == 'fd':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 1024]]
        first_stage_stride = True
    else:
        raise ValueError("Unsupported MobileNet version {}".format(version))

    channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        **kwargs)


def mobilenet_w1(**kwargs):
    return get_mobilenet('orig', 1, **kwargs)


def mobilenet_w3d4(**kwargs):
    return get_mobilenet('orig', 0.75, **kwargs)


def mobilenet_wd2(**kwargs):
    return get_mobilenet('orig', 0.5, **kwargs)


def mobilenet_wd4(**kwargs):
    return get_mobilenet('orig', 0.25, **kwargs)


def fdmobilenet_w1(**kwargs):
    return get_mobilenet('fd', 1, **kwargs)


def fdmobilenet_w3d4(**kwargs):
    return get_mobilenet('fd', 0.75, **kwargs)


def fdmobilenet_wd2(**kwargs):
    return get_mobilenet('fd', 0.5, **kwargs)


def fdmobilenet_wd4(**kwargs):
    return get_mobilenet('fd', 0.25, **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    global TESTING
    TESTING = True

    models = [
        mobilenet_w1,
        mobilenet_w3d4,
        mobilenet_wd2,
        mobilenet_wd4,
        fdmobilenet_w1,
        fdmobilenet_w3d4,
        fdmobilenet_wd2,
        fdmobilenet_wd4,
    ]

    for model in models:

        net = model()

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        assert (model != mobilenet_w1 or weight_count == 4231976)
        assert (model != mobilenet_w3d4 or weight_count == 2585560)
        assert (model != mobilenet_wd2 or weight_count == 1331592)
        assert (model != mobilenet_wd4 or weight_count == 470072)
        assert (model != fdmobilenet_w1 or weight_count == 2901288)
        assert (model != fdmobilenet_w3d4 or weight_count == 1833304)
        assert (model != fdmobilenet_wd2 or weight_count == 993928)
        assert (model != fdmobilenet_wd4 or weight_count == 383160)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

