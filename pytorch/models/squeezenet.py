"""
    SqueezeNet, implemented in PyTorch.
    Original paper: 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class FireConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding):
        super(FireConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class FireUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 squeeze_channels,
                 expand1x1_channels,
                 expand3x3_channels,
                 residual):
        super(FireUnit, self).__init__()
        self.residual = residual

        self.squeeze = FireConv(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            padding=0)
        self.expand1x1 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand1x1_channels,
            kernel_size=1,
            padding=0)
        self.expand3x3 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand3x3_channels,
            kernel_size=3,
            padding=1)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = torch.cat((y1, y2), dim=1)
        if self.residual:
            out = out + identity
        return out


class SqueezeInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(SqueezeInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class SqueezeNet(nn.Module):

    def __init__(self,
                 init_block_kernel_size,
                 init_block_channels,
                 residual,
                 in_channels=3,
                 num_classes=1000):
        super(SqueezeNet, self).__init__()
        channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
        residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]

        self.features = nn.Sequential()
        self.features.add_module("init_block", SqueezeInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=init_block_kernel_size))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            stage.add_module("pool{}".format(i + 1), nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True))
            for j in range(len(channels_per_stage)):
                out_channels = channels_per_stage[j]
                expand_channels = out_channels // 2
                squeeze_channels = out_channels // 8
                stage.add_module("fire{}".format(j + 1), FireUnit(
                    in_channels=in_channels,
                    squeeze_channels=squeeze_channels,
                    expand1x1_channels=expand_channels,
                    expand3x3_channels=expand_channels,
                    residual=(residual and residuals[i][j] == 1)))
                in_channels = out_channels
                self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('dropout', nn.Dropout(p=0.5))

        self.output = nn.Sequential()
        self.output.add_module('final_conv', nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1))
        self.output.add_module('final_activ', nn.ReLU(inplace=True))
        self.output.add_module('final_pool', nn.AvgPool2d(
            kernel_size=13,
            stride=1))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'final_conv' in name:
                    init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_squeezenet(version,
                   residual=False,
                   pretrained=False,
                   **kwargs):
    if version == '1.0':
        init_block_kernel_size = 7
        init_block_channels = 96
    elif version == '1.1':
        init_block_kernel_size = 3
        init_block_channels = 64
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return SqueezeNet(
        init_block_kernel_size=init_block_kernel_size,
        init_block_channels=init_block_channels,
        residual=residual,
        **kwargs)


def squeezenet1_0(**kwargs):
    return get_squeezenet('1.0', residual=False, **kwargs)


def squeezenet1_1(**kwargs):
    return get_squeezenet('1.1', residual=False, **kwargs)


def squeezeresnet1_0(**kwargs):
    return get_squeezenet(version='1.0', residual=True, **kwargs)


def squeezeresnet1_1(**kwargs):
    return get_squeezenet(version='1.1', residual=True, **kwargs)


def _test():
    import numpy as np
    from torch.autograd import Variable

    global TESTING
    TESTING = True

    net = squeezeresnet1_1()

    net.train()
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    #assert (weight_count == 1248424)  # squeezenet1_0
    #assert (weight_count == 1235496)  # squeezenet1_1
    #assert (weight_count == 1248424)  # squeezeresnet1_0
    assert (weight_count == 1235496)  # squeezeresnet1_1

    x = Variable(torch.randn(1, 3, 224, 224))
    y = net(x)
    assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()


