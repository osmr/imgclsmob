from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from .layers import ShuffleLayer, Conv, CondenseConv, CondenseLinear

__all__ = ['CondenseNet', 'oth_codensenet74_c4_g4', 'oth_codensenet74_c8_g8']


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = CondenseConv(in_channels, args.bottleneck * growth_rate,
                                   kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k-->k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):
    def __init__(self, args):

        super(CondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = CondenseLinear(self.num_features, args.num_classes,
                                         0.5)
        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def oth_codensenet74_c4_g4(**kwargs):
    from easydict import EasyDict
    args = EasyDict({'data': 'imagenet', 'stages': [4, 6, 8, 10, 8], 'growth': [8, 16, 32, 64, 128], 'group_1x1': 4,
                     'group_3x3': 4, 'condense_factor': 4, 'bottleneck': 4, 'num_classes': 1000})
    net = CondenseNet(args=args)
    return net


def oth_codensenet74_c8_g8(**kwargs):
    from easydict import EasyDict
    args = EasyDict({'data': 'imagenet', 'stages': [4, 6, 8, 10, 8], 'growth': [8, 16, 32, 64, 128], 'group_1x1': 8,
                     'group_3x3': 8, 'condense_factor': 8, 'bottleneck': 4, 'num_classes': 1000})
    net = CondenseNet(args=args)
    return net


if __name__ == "__main__":
    import numpy as np
    import torch
    from torch.autograd import Variable

    from easydict import EasyDict
    # args = EasyDict({'data': 'imagenet', 'stages': [4, 6, 8, 10, 8], 'growth': [8, 16, 32, 64, 128], 'group_1x1': 4,
    #                  'group_3x3': 4, 'condense_factor': 4, 'bottleneck': 4, 'num_classes': 1000})
    args = EasyDict({'data': 'imagenet', 'stages': [4, 6, 8, 10, 8], 'growth': [8, 16, 32, 64, 128], 'group_1x1': 8,
                     'group_3x3': 8, 'condense_factor': 8, 'bottleneck': 4, 'num_classes': 1000})
    net = CondenseNet(args=args)
    net = torch.nn.DataParallel(net)
    state_dict = torch.load("../imgclsmob_data/converted_condensenet_8.pth.tar", map_location='cpu')['state_dict']
    net.load_state_dict(state_dict)
    #torch.save(obj=net.cpu().module.state_dict(), f="../imgclsmob_data/converted_condensenet_8_pure.pth")

    input = Variable(torch.randn(1, 3, 224, 224))
    output = net(input)
    #print(output.size())
    #print("net={}".format(net))

    net.eval()
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    print("weight_count={}".format(weight_count))
