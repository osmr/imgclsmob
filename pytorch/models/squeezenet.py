"""
    SqueezeNet, implemented in Gluon.
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
        with self.name_scope():
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
                 expand3x3_channels):
        super(FireUnit, self).__init__()
        with self.name_scope():
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
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = torch.cat((y1, y2), dim=1)
        return out


def squeeze_pool():
    return nn.MaxPool2d(
        kernel_size=3,
        stride=2,
        ceil_mode=True)


class SqueezeInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(SqueezeInitBlock, self).__init__()
        with self.name_scope():
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
                 first_out_channels,
                 first_kernel_size,
                 pool_stages,
                 classes=1000):
        super(SqueezeNet, self).__init__()
        input_channels = 3
        stage_squeeze_channels = [16, 32, 48, 64]
        stage_expand_channels = [64, 128, 192, 256]

        with self.name_scope():
            self.features = nn.Sequential()
            self.features.add(SqueezeInitBlock(
                in_channels=input_channels,
                out_channels=first_out_channels,
                kernel_size=first_kernel_size))
            k = 0
            pool_ind = 0
            for i in range(len(stage_squeeze_channels)):
                for j in range(2):
                    if (pool_ind < len(pool_stages) - 1) and (k == pool_stages[pool_ind]):
                        self.features.add(squeeze_pool())
                        pool_ind += 1
                    in_channels = first_out_channels if (i == 0 and j == 0) else \
                        (2 * stage_expand_channels[i - 1] if j == 0 else 2 * stage_expand_channels[i])
                    self.features.add(FireUnit(
                        in_channels=in_channels,
                        squeeze_channels=stage_squeeze_channels[i],
                        expand1x1_channels=stage_expand_channels[i],
                        expand3x3_channels=stage_expand_channels[i]))
                    k += 1
            self.features.add(nn.Dropout(p=0.5))

            self.output = nn.Sequential()
            final_conv = nn.Conv2d(
                in_channels=(2 * stage_expand_channels[-1]),
                out_channels=classes,
                kernel_size=1)
            self.output.add(final_conv)
            self.output.add(nn.ReLU(inplace=True))
            self.output.add(nn.AvgPool2d(kernel_size=13))

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m is final_conv:
                        init.normal_(m.weight, mean=0.0, std=0.01)
                    else:
                        init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_squeezenet(version,
                   pretrained=False,
                   **kwargs):
    if version == '1.0':
        first_out_channels = 96
        first_kernel_size = 7
        pool_stages = [0, 3, 7]
    elif version == '1.1':
        first_out_channels = 64
        first_kernel_size = 3
        pool_stages = [0, 2, 4]
    else:
        raise ValueError("Unsupported SqueezeNet version {}: 1.0 or 1.1 expected".format(version))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    return SqueezeNet(
        first_out_channels=first_out_channels,
        first_kernel_size=first_kernel_size,
        pool_stages=pool_stages,
        **kwargs)


def squeezenet1_0(**kwargs):
    return get_squeezenet('1.0', **kwargs)


def squeezenet1_1(**kwargs):
    return get_squeezenet('1.1', **kwargs)

