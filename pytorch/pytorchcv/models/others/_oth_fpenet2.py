"""
    FPENet for image segmentation, implemented in PyTorch.
    Original paper: 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.
"""

__all__ = ['FPENet', 'fpenet_cityscapes']


import torch
import torch.nn as nn
import torch.nn.functional as F
from common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, SEBlock, InterpolationBlock


class FPEBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 downsample=None,
                 stride=1,
                 bottleneck_factor=1,
                 scales=4,
                 use_se=False):
        super(FPEBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.scales = scales
        self.use_se = use_se

        if in_channels % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        bottleneck_planes = in_channels * bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=bottleneck_planes,
            stride=stride)

        self.conv2 = nn.ModuleList([conv3x3_block(
            in_channels=bottleneck_planes // scales,
            out_channels=bottleneck_planes // scales,
            groups=(bottleneck_planes // scales),
            dilation=dilation[i],
            padding=dilation[i]) for i in range(scales)])

        self.conv3 = conv1x1_block(
            in_channels=bottleneck_planes,
            out_channels=out_channels,
            activation=None)

        if self.use_se:
            self.se = SEBlock(channels=out_channels)

        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        y = self.conv1(x)

        xs = torch.chunk(y, chunks=self.scales, dim=1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.conv2[s](xs[s]))
            else:
                ys.append(self.conv2[s](xs[s] + ys[-1]))
        y = torch.cat(ys, dim=1)

        y = self.conv3(y)

        if self.use_se:
            y = self.se(y)

        if self.downsample is not None:
            identity = self.downsample(identity)

        y += identity

        y = self.activ(y)

        return y


class MEUBlock(nn.Module):
    def __init__(self,
                 in_channels_high,
                 in_channels_low,
                 out_channels):
        super(MEUBlock, self).__init__()

        self.conv_low = conv1x1_block(
            in_channels=in_channels_low,
            out_channels=out_channels,
            activation=None)

        self.conv_high = conv1x1_block(
            in_channels=in_channels_high,
            out_channels=out_channels,
            activation=None)

        self.sa_conv = conv1x1(
            in_channels=1,
            out_channels=1)
        self.ca_conv = conv1x1(
            in_channels=out_channels,
            out_channels=out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w = fms_low.shape

        fms_low = self.conv_low(fms_low)
        fms_high = self.conv_high(fms_high)

        sa_avg_out = self.sigmoid(self.sa_conv(torch.mean(fms_low, dim=1, keepdim=True)))
        ca_avg_out = self.sigmoid(self.relu(self.ca_conv(self.avg_pool(fms_high))))

        fms_high_up = F.interpolate(fms_high, size=(h, w), mode='bilinear', align_corners=True)

        fms_sa_att = sa_avg_out * fms_high_up
        fms_ca_att = ca_avg_out * fms_low

        out = fms_ca_att + fms_sa_att

        return out


class FPENet(nn.Module):
    def __init__(self,
                 num_classes=19,
                 width=16,
                 scales=4,
                 use_se=False):
        super(FPENet, self).__init__()
        outplanes = [int(width * 2 ** i) for i in range(3)]  # planes=[16,32,64]

        self.block_num = [1, 3, 9]
        self.dilation = [1, 2, 4, 8]

        self.conv1 = conv3x3_block(
            in_channels=3,
            out_channels=outplanes[0],
            stride=2)
        self.inplanes = outplanes[0]

        self.layer1 = self._make_layer(outplanes[0], self.block_num[0], dilation=self.dilation,
                                       stride=1, bottleneck_factor=1, scales=scales, use_se=use_se)
        self.layer2 = self._make_layer(outplanes[1], self.block_num[1], dilation=self.dilation,
                                       stride=2, bottleneck_factor=4, scales=scales, use_se=use_se)
        self.layer3 = self._make_layer(outplanes[2], self.block_num[2], dilation=self.dilation,
                                       stride=2, bottleneck_factor=4, scales=scales, use_se=use_se)
        self.meu1 = MEUBlock(
            in_channels_high=64,
            in_channels_low=32,
            out_channels=64)
        self.meu2 = MEUBlock(
            in_channels_high=64,
            in_channels_low=16,
            out_channels=32)

        self.classifier = conv1x1(
            in_channels=32,
            out_channels=num_classes,
            bias=True)

        self.up = InterpolationBlock(
            scale_factor=2,
            align_corners=True)

    def _make_layer(self,
                    planes,
                    blocks,
                    dilation,
                    stride=1,
                    bottleneck_factor=1,
                    scales=4,
                    use_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = conv1x1_block(
                in_channels=self.inplanes,
                out_channels=planes,
                stride=stride,
                activation=None)

        layers = []
        layers.append(FPEBlock(
            in_channels=self.inplanes,
            out_channels=planes,
            dilation=dilation,
            downsample=downsample,
            stride=stride,
            bottleneck_factor=bottleneck_factor,
            scales=scales,
            use_se=use_se))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(FPEBlock(
                in_channels=self.inplanes,
                out_channels=planes,
                dilation=dilation,
                scales=scales,
                use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## stage 1
        x = self.conv1(x)

        x_1 = self.layer1(x)

        ## stage 2
        x_2_0 = self.layer2[0](x_1)
        x_2_1 = self.layer2[1](x_2_0)
        x_2_2 = self.layer2[2](x_2_1)
        x_2 = x_2_0 + x_2_2

        ## stage 3
        x_3_0 = self.layer3[0](x_2)
        x_3_1 = self.layer3[1](x_3_0)
        x_3_2 = self.layer3[2](x_3_1)
        x_3_3 = self.layer3[3](x_3_2)
        x_3_4 = self.layer3[4](x_3_3)
        x_3_5 = self.layer3[5](x_3_4)
        x_3_6 = self.layer3[6](x_3_5)
        x_3_7 = self.layer3[7](x_3_6)
        x_3_8 = self.layer3[8](x_3_7)
        x_3 = x_3_0 + x_3_8

        x2 = self.meu1(x_3, x_2)
        x1 = self.meu2(x2, x_1)

        y = self.classifier(x1)
        y = self.up(y)
        return y


def fpenet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    net = FPENet(num_classes=num_classes)
    return net


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False

    in_size = (1024, 2048)

    models = [
        fpenet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fpenet_cityscapes or weight_count == 115125)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
