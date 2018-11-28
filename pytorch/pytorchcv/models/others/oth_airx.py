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

__all__ = ['airx50_32x4d', 'airx101_32x4d_r16', 'airx101_32x4d_r2']


class AIRXBottleneck(nn.Module):
    """
    AIRXBottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, ratio=2, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width
            cardinality: num of convolution groups
            stride: conv stride. Replaces pooling layer
            ratio: dimensionality-compression ratio.
        """
        super(AIRXBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64.0)))
        C = cardinality
        self.stride = stride
        self.planes = planes

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if self.stride == 1 and self.planes < 512:  # for C2, C3, C4 stages
            self.conv_att1 = nn.Conv2d(inplanes, D * C // ratio, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_att1 = nn.BatchNorm2d(D * C // ratio)
            self.subsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.conv_att2 = nn.Conv2d(D*C // ratio, D*C // ratio, kernel_size=3, stride=2, padding=1, groups=C//2, bias=False)
            # self.bn_att2 = nn.BatchNorm2d(D*C // ratio)
            self.conv_att3 = nn.Conv2d(D * C // ratio, D * C // ratio, kernel_size=3, stride=1,
                                       padding=1, groups=C // ratio, bias=False)
            self.bn_att3 = nn.BatchNorm2d(D * C // ratio)
            self.conv_att4 = nn.Conv2d(D * C // ratio, D * C, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_att4 = nn.BatchNorm2d(D * C)
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


class AIRX(nn.Module):
    def __init__(self, baseWidth=4, cardinality=32, head7x7=True, ratio=2, layers=(3, 4, 23, 3), num_classes=1000):
        """ Constructor
        Args:
            baseWidth: baseWidth for AIRX.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(AIRX, self).__init__()
        block = AIRXBottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = 64

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, ratio)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, ratio)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, ratio)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
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
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, ratio, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, 1, ratio))

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


def airx50_32x4d():
    model = AIRX(baseWidth=4, cardinality=32, head7x7=False, layers=(3, 4, 6, 3), num_classes=1000)
    return model


def airx101_32x4d_r16():
    model = AIRX(baseWidth=4, cardinality=32, head7x7=False, ratio=16, layers=(3, 4, 23, 3), num_classes=1000)
    return model


def airx101_32x4d_r2():
    model = AIRX(baseWidth=4, cardinality=32, head7x7=False, layers=(3, 4, 23, 3), num_classes=1000)
    return model
