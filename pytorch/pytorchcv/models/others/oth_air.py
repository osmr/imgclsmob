from __future__ import division

""" 
Attention Inspiring Receptive-fields Network
Copyright (c) Yang Lu, 2018
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['oth_air50_1x64d', 'oth_air101_1x64d', 'oth_air50_1x64d_r16']


class AIRBottleneck(nn.Module):
    """
    AIRBottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, ratio=2, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer
            ratio: dimensionality-compression ratio.
        """
        super(AIRBottleneck, self).__init__()

        self.stride = stride
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if self.stride == 1 and self.planes < 512:  # for C2, C3, C4 stages
            self.conv_att1 = nn.Conv2d(inplanes, planes // ratio, kernel_size=1, stride=1, padding=0,  bias=False)
            self.bn_att1 = nn.BatchNorm2d(planes // ratio)
            self.subsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.conv_att2 = nn.Conv2d(planes // ratio, planes // ratio, kernel_size=3, stride=2, padding=1, bias=False)
            # self.bn_att2 = nn.BatchNorm2d(planes // ratio)
            self.conv_att3 = nn.Conv2d(planes // ratio, planes // ratio, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_att3 = nn.BatchNorm2d(planes // ratio)
            self.conv_att4 = nn.Conv2d(planes // ratio, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_att4 = nn.BatchNorm2d(planes)
            self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.stride == 1 and self.planes < 512:
            att = self.conv_att1(x)
            att = self.bn_att1(att)
            att = self.relu(att)
            # att = self.conv_att2(att)
            # att = self.bn_att2(att)
            # att = self.relu(att)
            att = self.subsample(att)
            att = self.conv_att3(att)
            att = self.bn_att3(att)
            att = self.relu(att)
            att = F.upsample(att, size=out.size()[2:], mode='bilinear')
            att = self.conv_att4(att)
            att = self.bn_att4(att)
            att = self.sigmoid(att)
            out = out * att

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AIR(nn.Module):
    def __init__(self, baseWidth=64, head7x7=True, ratio=2, layers=(3, 4, 23, 3), num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., [3, 4, 23, 3]
            num_classes: number of classes
        """
        super(AIR, self).__init__()
        block = AIRBottleneck

        self.inplanes = baseWidth

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, baseWidth, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth)
        else:
            self.conv1 = nn.Conv2d(3, baseWidth // 2, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth // 2)
            self.conv2 = nn.Conv2d(baseWidth // 2, baseWidth // 2, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(baseWidth // 2)
            self.conv3 = nn.Conv2d(baseWidth // 2, baseWidth, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(baseWidth)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, baseWidth, layers[0], 1, ratio)
        self.layer2 = self._make_layer(block, baseWidth * 2, layers[1], 2, ratio)
        self.layer3 = self._make_layer(block, baseWidth * 4, layers[2], 2, ratio)
        self.layer4 = self._make_layer(block, baseWidth * 8, layers[3], 2, ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(baseWidth * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ratio=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct AIR
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, ratio, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def oth_air50_1x64d(pretrained=False, **kwargs):
    model = AIR(baseWidth=64, head7x7=False, layers=(3, 4, 6, 3), num_classes=1000)
    return model


def oth_air50_1x64d_r16(pretrained=False, **kwargs):
    model = AIR(baseWidth=64, head7x7=False, layers=(3, 4, 6, 3), num_classes=1000, ratio=16)
    return model


def oth_air101_1x64d(pretrained=False, **kwargs):
    model = AIR(baseWidth=64, head7x7=False, layers=(3, 4, 23, 3), num_classes=1000)
    return model


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        oth_air50_1x64d,
        oth_air101_1x64d,
        oth_air50_1x64d_r16,
    ]

    for model in models:

        net = model()

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_air50_1x64d or weight_count == 27425864)
        assert (model != oth_air101_1x64d or weight_count == 51727432)
        assert (model != oth_air50_1x64d_r16 or weight_count == 25714952)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
