import sys
import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models


class Bottleneck(nn.Sequential):
    def __init__(self, nChannels, growthRate, dropRate=0.0):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        if dropRate > 0:
            # test inplace
            self.dropout1 = nn.Dropout2d(dropRate)

        self.bn2 = nn.BatchNorm2d(interChannels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
        if dropRate > 0:
            # test inplace
            self.dropout2 = nn.Dropout2d(dropRate)


class SingleLayer(nn.Sequential):
    def __init__(self, nChannels, growthRate, dropRate=0.0):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        if dropRate > 0:
            # test inplace
            self.dropout1 = nn.Dropout2d(dropRate)


class BottleneckWithConcat(Bottleneck):
    def __init__(self, nChannels, growthRate, dropRate=0.0):
        super(Bottleneck, self).__init__(nChannels, growthRate, dropRate)

    def forward(self, x):
        out = super(Bottleneck, self).forward(x)
        out = torch.cat((x, out), 1)
        return out


class SingleLayerWithConcat(SingleLayer):
    def __init__(self, nChannels, growthRate, dropRate=0.0):
        super(SingleLayerWithConcat, self).__init__(nChannels, growthRate, dropRate)

    def forward(self, x):
        out = super(SingleLayer, self).forward(x)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Sequential):
    def __init__(self, nChannels, nOutChannels, dropRate=0.0):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        if dropRate > 0.0:
            self.dropout1 = nn.Dropout2d(dropRate)
        self.avgpool = nn.AvgPool2d(2)


def linearFetch(lst):
    return lst


def expFetch(lst):
    nums = len(lst)
    i = 1
    res = []
    while i <= nums:
        res.append(lst[nums - i])
        i *= 2
    return res


class DenseStage(nn.Sequential):
    def __init__(self, nDenseBlocks, nChannels, growthRate, bottleneck, dropRate=0.0, fetch="linear"):
        super(DenseStage, self).__init__()

        self.nDenseBlocks = nDenseBlocks
        self.prevChannels = [nChannels]
        if fetch in ["linear", "dense"]:
            self.fetch = linearFetch
        elif fetch in ["exp", "sparse"]:
            self.fetch = expFetch
        else:
            raise NotImplementedError("fetch %s is not supported" % fetch)

        for i in range(int(nDenseBlocks)):
            nChannels = sum(self.fetch(self.prevChannels))
            if bottleneck:
                unit = Bottleneck(nChannels, growthRate, dropRate)
            else:
                unit = SingleLayer(nChannels, growthRate, dropRate)
            self.add_module("block-%d" % (i + 1), unit)
            self.prevChannels.append(growthRate)
            # nChannels += growthRate

        self.nOutChannels = sum(self.fetch(self.prevChannels))

    def forward(self, x):
        # return super(DenseStage, self).forward(x)
        prev_outputs = [x]
        for i in range(int(self.nDenseBlocks)):
            out = self._modules["block-%d" % (i + 1)](x)
            # x = torch.cat([x, out], 1)
            prev_outputs.append(out)
            fetch_outputs = self.fetch(prev_outputs)
            x = torch.cat(fetch_outputs, 1).contiguous()
        return x


class Flattern(nn.Module):
    def __init__(self, dim=0):
        super(Flattern, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view(x.size(self.dim), -1)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'dim=' + str(self.dim) + ')'


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, layer_per_stage=None, grate_per_stage=None,
                 dropRate=0.0, fetch="exp"):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        else:
            reduction = 1
        if layer_per_stage is None:
            layer_per_stage = [nDenseBlocks, ] * 3
        if grate_per_stage is None:
            grate_per_stage = [growthRate, ] * 3

        if fetch in ["linear", "dense"]:
            nChannels = 2 * growthRate
        elif fetch in ["exp", "sparse"]:
            nChannels = growthRate
        else:
            raise NotImplementedError

        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)),
        ]))

        # self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)

        for i, layers in enumerate(layer_per_stage):
            stage = DenseStage(layer_per_stage[i], nChannels, grate_per_stage[i], bottleneck, dropRate=dropRate,
                               fetch=fetch)
            nChannels = stage.nOutChannels
            self.features.add_module("dense-stage-%d" % i, stage)
            if i < len(layer_per_stage) - 1:
                nOutChannels = int(math.floor(nChannels * reduction))
                trans = Transition(nChannels, nOutChannels)
                self.features.add_module("transition-%d" % i, trans)
                nChannels = nOutChannels
        self.features.add_module("norm_last", nn.BatchNorm2d(nChannels))
        self.features.add_module("relu_last", nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(OrderedDict([
            ("avgpool", nn.AvgPool2d(kernel_size=8)),
            ("flattern", Flattern(dim=0)),
            ("linear", nn.Linear(nChannels, nClasses))
        ]))

        self.init_weights()

    def init_weights(self):
        # adapt from torch setting
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


if __name__ == "__main__":
    depth = 100
    growthRate = 12

    bottleneck = True

    torch.manual_seed(0)
    np.random.seed(0)

    net = DenseNet(growthRate=12, depth=40, nClasses=100, reduction=0.5, bottleneck=False, grate_per_stage=None,
                   fetch="linear")

    print(net)
    total = sum([p.data.nelement() for p in net.parameters()])
    print('  + Number of params: %.2f' % (total / 1e6))

    # from visualize import make_dot
    sample_input = torch.ones([12, 3, 32, 32])
    sample_input = Variable(sample_input)

    output = net(sample_input)
    print(output.size(), output.sum())
