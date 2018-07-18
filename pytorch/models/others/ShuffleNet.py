import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .common import channel_shuffle

__all__ = [
    'shufflenet_group_1',
    'shufflenet_group_2',
    'shufflenet_group_3',
    'shufflenet_group_4',
    'shufflenet_group_8',
    'oth_shufflenet1_0_g1',
    'oth_shufflenet1_0_g2',
    'oth_shufflenet1_0_g3',
    'oth_shufflenet1_0_g4',
    'oth_shufflenet1_0_g8',
    'oth_shufflenet0_5_g3'
]


def depthwise_conv(c, stride):
    return nn.Conv2d(c, c, 3, stride=stride, padding=1, groups=c, bias=False)


def group_conv(in_c, out_c, groups):
    return nn.Conv2d(in_c, out_c, 1, groups=groups, bias=False)


class ShuffleUnit(nn.Module):
    def __init__(self, in_c, out_c, downsample, groups, ignore_group):
        super(ShuffleUnit, self).__init__()
        bott = out_c // 4
        self.downsample = downsample
        self.groups = groups
        if downsample:
            out_c -= in_c
            if ignore_group:
                self.compress = group_conv(in_c, bott, 1)
            else:
                self.compress = group_conv(in_c, bott, groups)
            self.bn_compress = nn.BatchNorm2d(bott)
            self.depthwise = depthwise_conv(bott, 2)
            self.bn_depthwise = nn.BatchNorm2d(bott)
            self.expand = group_conv(bott, out_c, groups)
            self.bn_expand = nn.BatchNorm2d(out_c)
            self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        else:
            self.compress = group_conv(in_c, bott, groups)
            self.bn_compress = nn.BatchNorm2d(bott)
            self.depthwise = depthwise_conv(bott, 1)
            self.bn_depthwise = nn.BatchNorm2d(bott)
            self.expand = group_conv(bott, out_c, groups)
            self.bn_expand = nn.BatchNorm2d(out_c)

    def forward(self, x):
        residual = x
        x = self.compress(x)
        x = self.bn_compress(x)
        x = F.relu(x, inplace=True)
        x = channel_shuffle(x, self.groups)
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.expand(x)
        x = self.bn_expand(x)
        if self.downsample:
            residual = self.pool(residual)
            x = torch.cat((x, residual), dim=1)
        else:
            x += residual
        x = F.relu(x, inplace=True)
        return x


class ShuffleNet(nn.Module):
    def __init__(self, block_channels, block_layers, init_channels, groups):
        super(ShuffleNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(3, init_channels, 3, stride=2, padding=1, bias=False)),
            ('norm_1', nn.BatchNorm2d(init_channels)),
            ('relu_1', nn.ReLU(inplace=True)),
            ('pool_1', nn.MaxPool2d(3, stride=2, padding=1)),
        ]))
        in_c = init_channels
        for i, (out_c, num_layers) in enumerate(zip(block_channels, block_layers)):
            self.features.add_module(
                'stage_{}_{}'.format(i + 1, 1),
                ShuffleUnit(in_c, out_c, True, groups, (i == 0))
            )
            for _ in range(num_layers):
                self.features.add_module(
                    'stage_{}_{}'.format(i + 1, _ + 2),
                    ShuffleUnit(out_c, out_c, False, groups, False)
                )
            in_c = out_c
        self.avg_pool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(in_c, 1000)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def shufflenet_group_1(model_config):
    init_mul = model_config['init_mul']
    width_mul = model_config['width_mul']
    init_channels = int(24 * init_mul)
    block_channels = [144, 288, 576]
    block_channels = [int(x * width_mul) for x in block_channels]
    block_layers = [3, 7, 3]
    return ShuffleNet(block_channels, block_layers, init_channels, 1)


def shufflenet_group_2(model_config):
    init_mul = model_config['init_mul']
    width_mul = model_config['width_mul']
    init_channels = (int(24 * init_mul) + 1) // 2 * 2
    block_channels = [200, 400, 800]
    block_channels = [(int(x * width_mul) + 1) // 2 * 2 for x in block_channels]
    block_layers = [3, 7, 3]
    return ShuffleNet(block_channels, block_layers, init_channels, 2)


def shufflenet_group_3(model_config):
    init_mul = model_config['init_mul']
    width_mul = model_config['width_mul']
    init_channels = (int(24 * init_mul) + 2) // 3 * 3
    block_channels = [240, 480, 960]
    block_channels = [(int(x * width_mul) + 2) // 3 * 3 for x in block_channels]
    block_layers = [3, 7, 3]
    return ShuffleNet(block_channels, block_layers, init_channels, 3)


def shufflenet_group_4(model_config):
    init_mul = model_config['init_mul']
    width_mul = model_config['width_mul']
    init_channels = (int(24 * init_mul) + 3) // 4 * 4
    block_channels = [272, 544, 1088]
    block_channels = [(int(x * width_mul) + 3) // 4 * 4 for x in block_channels]
    block_layers = [3, 7, 3]
    return ShuffleNet(block_channels, block_layers, init_channels, 4)


def shufflenet_group_8(model_config):
    init_mul = model_config['init_mul']
    width_mul = model_config['width_mul']
    init_channels = (int(24 * init_mul) + 7) // 8 * 8
    block_channels = [384, 768, 1536]
    block_channels = [(int(x * width_mul) + 7) // 8 * 8 for x in block_channels]
    block_layers = [3, 7, 3]
    return ShuffleNet(block_channels, block_layers, init_channels, 8)


def oth_shufflenet1_0_g1(**kwargs):
    return shufflenet_group_1({"init_mul": 1, 'width_mul': 1})


def oth_shufflenet1_0_g2(**kwargs):
    return shufflenet_group_2({"init_mul": 1, 'width_mul': 1})


def oth_shufflenet1_0_g3(**kwargs):
    return shufflenet_group_3({"init_mul": 1, 'width_mul': 1})


def oth_shufflenet1_0_g4(**kwargs):
    return shufflenet_group_4({"init_mul": 1, 'width_mul': 1})


def oth_shufflenet1_0_g8(**kwargs):
    return shufflenet_group_8({"init_mul": 1, 'width_mul': 1})


def oth_shufflenet0_5_g3(**kwargs):
    return shufflenet_group_3({"init_mul": 0.5, 'width_mul': 0.5})

