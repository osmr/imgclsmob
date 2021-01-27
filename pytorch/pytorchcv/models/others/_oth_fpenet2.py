"""
    FPENet for image segmentation, implemented in PyTorch.
    Original paper: 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.
"""

__all__ = ["oth_fpenet_cityscapes"]


import torch
import torch.nn as nn
import torch.nn.functional as F
from common import conv1x1, conv3x3, SEBlock


class FPEBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilat,
                 downsample=None,
                 stride=1,
                 t=1,
                 scales=4,
                 use_se=False,
                 norm_layer=None):
        super(FPEBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.scales = scales
        self.use_se = use_se

        if in_channels % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = in_channels * t

        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=bottleneck_planes,
            stride=stride)
        self.bn1 = norm_layer(bottleneck_planes)

        self.conv2 = nn.ModuleList([conv3x3(
            in_channels=bottleneck_planes // scales,
            out_channels=bottleneck_planes // scales,
            groups=(bottleneck_planes // scales),
            dilation=dilat[i],
            padding=1*dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])

        self.conv3 = conv1x1(
            in_channels=bottleneck_planes,
            out_channels=out_channels)
        self.bn3 = norm_layer(out_channels)

        if self.use_se:
            self.se = SEBlock(channels=out_channels)

        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.activ(y)

        xs = torch.chunk(y, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.activ(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.activ(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))
        y = torch.cat(ys, 1)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.use_se:
            y = self.se(y)

        if self.downsample is not None:
            identity = self.downsample(identity)

        y += identity

        y = self.activ(y)

        return y


class MEUModule(nn.Module):
    def __init__(self,
                 channels_high,
                 channels_low,
                 channel_out):
        super(MEUModule, self).__init__()

        self.conv1x1_low = nn.Conv2d(
            in_channels=channels_low,
            out_channels=channel_out,
            kernel_size=1,
            bias=False)

        self.bn_low = nn.BatchNorm2d(channel_out)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        self.conv1x1_high = nn.Conv2d(channels_high, channel_out, kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(channel_out)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(channel_out, channel_out, kernel_size=1, bias=False)

        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w = fms_low.shape

        #
        fms_low = self.conv1x1_low(fms_low)
        fms_low = self.bn_low(fms_low)
        sa_avg_out = self.sa_sigmoid(self.sa_conv(torch.mean(fms_low, dim=1, keepdim=True)))

        #
        fms_high = self.conv1x1_high(fms_high)
        fms_high = self.bn_high(fms_high)
        ca_avg_out = self.ca_sigmoid(self.relu(self.ca_conv(self.avg_pool(fms_high))))

        #
        fms_high_up = F.interpolate(fms_high, size=(h,w), mode='bilinear', align_corners=True)
        fms_sa_att = sa_avg_out * fms_high_up
        #
        fms_ca_att = ca_avg_out * fms_low

        out = fms_ca_att + fms_sa_att

        return out


class FPENet(nn.Module):
    def __init__(self,
                 num_classes=19,
                 width=16,
                 scales=4,
                 use_se=False,
                 norm_layer=None):
        super(FPENet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        outplanes = [int(width * 2 ** i) for i in range(3)]  # planes=[16,32,64]

        self.block_num = [1, 3, 9]
        self.dilation = [1, 2, 4, 8]

        self.inplanes = outplanes[0]
        self.conv1 = nn.Conv2d(3, outplanes[0], kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1 = norm_layer(outplanes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FPEBlock, outplanes[0], self.block_num[0], dilation=self.dilation,
                                       stride=1, t=1, scales=scales, use_se=use_se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(FPEBlock, outplanes[1], self.block_num[1], dilation=self.dilation,
                                       stride=2, t=4, scales=scales, use_se=use_se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(FPEBlock, outplanes[2], self.block_num[2], dilation=self.dilation,
                                       stride=2, t=4, scales=scales, use_se=use_se, norm_layer=norm_layer)
        self.meu1 = MEUModule(64, 32, 64)
        self.meu2 = MEUModule(64, 16, 32)

        # Projection layer
        self.project_layer = nn.Conv2d(32, num_classes, kernel_size = 1)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    dilation,
                    stride=1,
                    t=1,
                    scales=4,
                    use_se=False,
                    norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(
            self.inplanes,
            planes,
            dilat=dilation,
            downsample=downsample,
            stride=stride,
            t=t,
            scales=scales,
            use_se=use_se,
            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes,
                planes,
                dilat=dilation,
                scales=scales,
                use_se=use_se,
                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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

        output = self.project_layer(x1)

        # Bilinear interpolation x2
        output = F.interpolate(output,scale_factor=2, mode='bilinear', align_corners=True)

        return output


def oth_fpenet_cityscapes(num_classes=19, pretrained=False, **kwargs):
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
        oth_fpenet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_fpenet_cityscapes or weight_count == 115125)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
