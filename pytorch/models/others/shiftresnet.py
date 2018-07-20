"""
    ShiftResNet.
    Original paper: 'Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions'
    Source repo: https://github.com/alvinwan/shiftresnet-cifar
"""

"""PyTorch implementation of ShiftResNet

ShiftResNet modifications written by Bichen Wu and Alvin Wan. Efficient CUDA
implementation of shift written by Peter Jin.

Reference:
[1] Bichen Wu, Alvin Wan, Xiangyu Yue, Peter Jin, Sicheng Zhao, Noah Golmant,
    Amir Gholaminejad, Joseph Gonzalez, Kurt Keutzer
    Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions.
    arXiv:1711.08141
"""

import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet
from models.shiftnet_cuda_v2.nn import GenericShift_cuda


class ShiftConv(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(ShiftConv, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = mid_planes = int(out_planes * self.expansion)

        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.shift2 = GenericShift_cuda(kernel_size=3, dilate_factor=1)
        self.conv2 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                  in_planes, out_planes, kernel_size=1, stride=stride,
                  bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.in_planes * self.mid_planes + \
                out_h * out_w * self.mid_planes * self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = x.size()
        x = F.relu(self.bn2(self.conv2(self.shift2(x))))
        self.out_nchw = x.size()
        x += shortcut
        return x


def ShiftResNet20(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [3, 3, 3], num_classes=num_classes)


def ShiftResNet56(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [9, 9, 9], num_classes=num_classes)


def ShiftResNet110(expansion=1, num_classes=10):
    block = lambda in_planes, out_planes, stride: \
        ShiftConv(in_planes, out_planes, stride, expansion=expansion)
    return ResNet(block, [18, 18, 18], num_classes=num_classes)

