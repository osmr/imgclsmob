"""
    MobileNetV2, implemented in PyTorch.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
"""

__all__ = ['MobileNetV2', 'mobilenetv2_w1', 'mobilenetv2_w3d4', 'mobilenetv2_wd2', 'mobilenetv2_wd4']

import torch.nn as nn
import torch.nn.init as init


class MobnetConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 activate):
        super(MobnetConv, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            self.activ = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def mobnet_conv1x1(in_channels,
                   out_channels,
                   activate):
    return MobnetConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        activate=activate)


def mobnet_dwconv3x3(in_channels,
                     out_channels,
                     stride,
                     activate):
    return MobnetConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=out_channels,
        activate=activate)


class LinearBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion):
        super(LinearBottleneck, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels

        self.conv1 = mobnet_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            activate=True)
        self.conv2 = mobnet_dwconv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activate=True)
        self.conv3 = mobnet_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            activate=False)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 num_classes=1000):
        super(MobileNetV2, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", MobnetConv(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            activate=True))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                stage.add_module("unit{}".format(j + 1), LinearBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', mobnet_conv1x1(
            in_channels=in_channels,
            out_channels=final_block_channels,
            activate=True))
        in_channels = final_block_channels
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_mobilenetv2(width_scale,
                    pretrained=False,
                    **kwargs):
    """
    Create MobileNetV2 model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [[]])

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = int(final_block_channels * width_scale)

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return MobileNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)


def mobilenetv2_w1(**kwargs):
    """
    1.0 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    return get_mobilenetv2(1.0, **kwargs)


def mobilenetv2_w3d4(**kwargs):
    """
    0.75 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    return get_mobilenetv2(0.75, **kwargs)


def mobilenetv2_wd2(**kwargs):
    """
    0.5 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    return get_mobilenetv2(0.5, **kwargs)


def mobilenetv2_wd4(**kwargs):
    """
    0.25 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    return get_mobilenetv2(0.25, **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    models = [
        mobilenetv2_w1,
        mobilenetv2_w3d4,
        mobilenetv2_wd2,
        mobilenetv2_wd4,
    ]

    for model in models:

        net = model()

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        assert (model != mobilenetv2_w1 or weight_count == 3504960)
        assert (model != mobilenetv2_w3d4 or weight_count == 2627592)
        assert (model != mobilenetv2_wd2 or weight_count == 1964736)
        assert (model != mobilenetv2_wd4 or weight_count == 1516392)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

