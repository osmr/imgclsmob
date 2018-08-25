"""
    NASNet-A Mobile, implemented in PyTorch.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.
"""

__all__ = ['nasnet_a_mobile']

import torch
import torch.nn as nn
import torch.nn.init as init
from common import conv1x1


class DualPathSequential(nn.Sequential):

    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 *args):
        super(DualPathSequential, self).__init__(*args)
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals

    def forward(self, x, x_prev=None):
        length = len(self._modules.values())
        for i, module in enumerate(self._modules.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x, x_prev = module(x), x
            else:
                x, x_prev = module(x, x_prev)
        if self.return_two:
            return x, x_prev
        else:
            return x


def nasnet_batch_norm(channels):
    return nn.BatchNorm2d(
        num_features=channels,
        eps=0.001,
        momentum=0.1,
        affine=True)


def nasnet_maxpool():
    return nn.MaxPool2d(
        kernel_size=3,
        stride=2,
        padding=1)


def nasnet_avgpool1x1_s2():
    return nn.AvgPool2d(
        kernel_size=1,
        stride=2,
        count_include_pad=False)


def nasnet_avgpool3x3_s1():
    return nn.AvgPool2d(
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False)


def nasnet_avgpool3x3_s2():
    return nn.AvgPool2d(
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False)


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.pool = nasnet_maxpool()

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class AvgPoolPad(nn.Module):

    def __init__(self,
                 stride=2,
                 padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=stride,
            padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class NasConv(nn.Module):
    """
    NASNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups):
        super(NasConv, self).__init__()
        self.activ = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def nas_conv1x1(in_channels,
                out_channels):
    """
    1x1 version of the NASNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return NasConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1)


class DwsConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=False):
        super(DwsConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias)
        self.pw_conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class NasDwsConv(nn.Module):
    """
    NASNet specific DWS convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 specific=False):
        super(NasDwsConv, self).__init__()
        self.specific = specific

        self.activ = nn.ReLU()
        self.conv = DwsConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nasnet_batch_norm(channels=out_channels)
        if self.specific:
            self.padding = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        x = self.activ(x)
        if self.specific:
            x = self.padding(x)
        x = self.conv(x)
        if self.specific:
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn(x)
        return x


class DwsBranch(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 specific=False,
                 stem=False):
        super(DwsBranch, self).__init__()
        assert (not stem) or (not specific)
        mid_channels = out_channels if stem else in_channels

        self.conv1 = NasDwsConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            specific=specific)
        self.conv2 = NasDwsConv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def dws_branch_k3_s1_p1(in_channels,
                        out_channels,
                        specific=False):
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        specific=specific)


def dws_branch_k5_s1_p2(in_channels,
                        out_channels,
                        specific=False):
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=1,
        padding=2,
        specific=specific)


def dws_branch_k5_s2_p2(in_channels,
                        out_channels,
                        specific=False,
                        stem=False):
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=2,
        padding=2,
        specific=specific,
        stem=stem)


def dws_branch_k7_s2_p3(in_channels,
                        out_channels,
                        specific=False,
                        stem=False):
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=2,
        padding=3,
        specific=specific,
        stem=stem)


class NasPathBranch(nn.Module):
    """
    NASNet specific `path-branch` block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 specific=False):
        super(NasPathBranch, self).__init__()
        self.specific = specific

        self.avgpool = nasnet_avgpool1x1_s2()
        self.conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)
        if self.specific:
            self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))

    def forward(self, x):
        if self.specific:
            x = self.padding(x)
            x = x[:, :, 1:, 1:].contiguous()
        x = self.avgpool(x)
        x = self.conv(x)
        return x


class NasPathBlock(nn.Module):
    """
    NASNet specific `path-branch` block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(NasPathBlock, self).__init__()
        mid_channels = out_channels // 2

        self.activ = nn.ReLU()
        self.path1 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.path2 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels,
            specific=True)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.activ(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        return x


class Stem1Unit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(Stem1Unit, self).__init__()
        mid_channels = out_channels // 4

        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True)

        self.comb1_left = nasnet_maxpool()
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True)

        self.comb2_left = nasnet_avgpool3x3_s2()
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb4_right = nasnet_maxpool()

    def forward(self, x, _=None):
        x_left = self.conv1x1(x)
        x_right = x

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        return x_out, x


class Stem2Unit(nn.Module):

    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels):
        super(Stem2Unit, self).__init__()
        mid_channels = out_channels // 4

        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.path = NasPathBlock(
            in_channels=prev_in_channels,
            out_channels=mid_channels)

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)

        self.comb1_left = MaxPoolPad()
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)

        self.comb2_left = AvgPoolPad()
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)
        self.comb4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        return x_out, x


class FirstUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels):
        super(FirstUnit, self).__init__()
        mid_channels = out_channels // 6

        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.path = NasPathBlock(
            in_channels=prev_in_channels,
            out_channels=mid_channels)

        self.comb0_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb0_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

        self.comb1_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb1_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

        self.comb2_left = nasnet_avgpool3x3_s1()

        self.comb3_left = nasnet_avgpool3x3_s1()
        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
        x4 = self.comb4_left(x_left) + x_left

        x_out = torch.cat((x_right, x0, x1, x2, x3, x4), dim=1)
        return x_out, x


class NormalUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels):
        super(NormalUnit, self).__init__()
        mid_channels = out_channels // 6

        self.conv1x1_prev = nas_conv1x1(
            in_channels=prev_in_channels,
            out_channels=mid_channels)
        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.comb0_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb0_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

        self.comb1_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb1_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

        self.comb2_left = nasnet_avgpool3x3_s1()

        self.comb3_left = nasnet_avgpool3x3_s1()
        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels)

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
        x4 = self.comb4_left(x_left) + x_left

        x_out = torch.cat((x_right, x0, x1, x2, x3, x4), dim=1)
        return x_out, x


class ReductionUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels):
        super(ReductionUnit, self).__init__()
        mid_channels = out_channels // 4

        self.conv1x1_prev = nas_conv1x1(
            in_channels=prev_in_channels,
            out_channels=mid_channels)
        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)

        self.comb1_left = MaxPoolPad()
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)

        self.comb2_left = AvgPoolPad()
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            specific=True)
        self.comb4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_next = torch.cat((x1, x2, x3, x4), dim=1)
        return x_next, x


class NASNetInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(NASNetInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class NASNet(nn.Module):

    def __init__(self,
                 init_block_channels,
                 stem_blocks_channels,
                 channels,
                 in_channels=3,
                 num_classes=1000):
        super(NASNet, self).__init__()

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=2)
        self.features.add_module("init_block", NASNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels

        out_channels = stem_blocks_channels[0]
        self.features.add_module("stem1_unit", Stem1Unit(
            in_channels=in_channels,
            out_channels=out_channels))
        prev_in_channels = in_channels
        in_channels = out_channels

        out_channels = stem_blocks_channels[1]
        self.features.add_module("stem2_unit", Stem2Unit(
            in_channels=in_channels,
            prev_in_channels=prev_in_channels,
            out_channels=out_channels))
        prev_in_channels = in_channels
        in_channels = out_channels

        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = ReductionUnit
                elif ((i == 0) and (j == 0)) or ((i != 0) and (j == 1)):
                    unit = FirstUnit
                else:
                    unit = NormalUnit
                stage.add_module("unit{}".format(j + 1), unit(
                    in_channels=in_channels,
                    prev_in_channels=prev_in_channels,
                    out_channels=out_channels))
                prev_in_channels = in_channels
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("activ", nn.ReLU())
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Sequential()
        self.output.add_module('dropout', nn.Dropout(p=0.5))
        self.output.add_module('fc', nn.Linear(
            in_features=in_channels,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_nasnet(repeat,
               penultimate_filters,
               pretrained=False,
               **kwargs):

    init_block_channels = 32
    stem_blocks_channels = [1, 2]
    channels = [[6, 6, 6, 6],
                [8, 12, 12, 12, 12],
                [16, 24, 24, 24, 24]]
    base_channel_chunk = penultimate_filters // channels[-1][-1]

    stem_blocks_channels = [(ci * base_channel_chunk) for ci in stem_blocks_channels]
    channels = [[(cij * base_channel_chunk) for cij in ci] for ci in channels]

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = NASNet(
        init_block_channels=init_block_channels,
        stem_blocks_channels=stem_blocks_channels,
        channels=channels,
        **kwargs)
    return net


def nasnet_a_mobile(**kwargs):
    return get_nasnet(repeat=4, penultimate_filters=1056, **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        nasnet_a_mobile,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nasnet_a_mobile or weight_count == 5289978)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

