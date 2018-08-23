"""
    NASNet-A Mobile, implemented in PyTorch.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.
"""

__all__ = ['nasnet_a_mobile']

import torch
import torch.nn as nn


class DoubleLinkedSequential(nn.Sequential):

    def __init__(self, *args):
        super(DoubleLinkedSequential, self).__init__(*args)

    def forward(self, x, x_prev=None):
        for module in self._modules.values():
            if x_prev is None:
                x_prev, x = x, module(x)
            else:
                x_prev, x = x, module(x, x_prev)
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


def conv1x1(in_channels,
            out_channels,
            stride=1,
            bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias)


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
    activate : bool
        Whether activate the convolution block.
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


class CellStem0(nn.Module):
    def __init__(self,
                 stem_filters,
                 num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters

        self.conv_1x1 = nas_conv1x1(
            in_channels=self.stem_filters,
            out_channels=self.num_filters)

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=self.num_filters,
            out_channels=self.num_filters)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=self.stem_filters,
            out_channels=self.num_filters,
            stem=True)

        self.comb1_left = nasnet_maxpool()
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=self.stem_filters,
            out_channels=self.num_filters,
            stem=True)

        self.comb2_left = nasnet_avgpool3x3_s2()
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=self.stem_filters,
            out_channels=self.num_filters,
            stem=True)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=self.num_filters,
            out_channels=self.num_filters)
        self.comb4_right = nasnet_maxpool()

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb0_left = self.comb0_left(x1)
        x_comb0_right = self.comb0_right(x)
        x_comb0 = x_comb0_left + x_comb0_right

        x_comb1_left = self.comb1_left(x1)
        x_comb1_right = self.comb1_right(x)
        x_comb1 = x_comb1_left + x_comb1_right

        x_comb2_left = self.comb2_left(x1)
        x_comb2_right = self.comb2_right(x)
        x_comb2 = x_comb2_left + x_comb2_right

        x_comb3_right = self.comb3_right(x_comb0)
        x_comb3 = x_comb3_right + x_comb1

        x_comb4_left = self.comb4_left(x_comb0)
        x_comb4_right = self.comb4_right(x1)
        x_comb4 = x_comb4_left + x_comb4_right

        x_out = torch.cat([x_comb1, x_comb2, x_comb3, x_comb4], 1)
        return x_out


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

        self.relu = nn.ReLU()
        self.path1 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.path2 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels,
            specific=True)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.relu(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = torch.cat([x1, x2], 1)
        x = self.bn(x)
        return x


class CellStem1(nn.Module):

    def __init__(self,
                 stem_filters,
                 num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters

        self.conv_1x1 = nas_conv1x1(
            in_channels=2*self.num_filters,
            out_channels=self.num_filters)

        self.path = NasPathBlock(
            in_channels=self.stem_filters,
            out_channels=self.num_filters)
        # self.relu = nn.ReLU()
        #
        # self.path_1 = NasPathBranch(
        #     in_channels=self.stem_filters,
        #     out_channels=self.num_filters//2)
        # # self.path_1 = nn.Sequential()
        # # self.path_1.add_module('avgpool', nasnet_avgpool1x1_s2())
        # # self.path_1.add_module('conv', conv1x1(
        # #     in_channels=self.stem_filters,
        # #     out_channels=self.num_filters//2))
        #
        # self.path_2 = NasPathBranch(
        #     in_channels=self.stem_filters,
        #     out_channels=self.num_filters//2,
        #     specific=True)
        # # self.path_2 = nn.ModuleList()
        # # self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        # # self.path_2.add_module('avgpool', nasnet_avgpool1x1_s2())
        # # self.path_2.add_module('conv', conv1x1(
        # #     in_channels=self.stem_filters,
        # #     out_channels=self.num_filters//2))
        #
        # self.final_path_bn = nasnet_batch_norm(channels=self.num_filters)

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            specific=True)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            specific=True)

        self.comb1_left = MaxPoolPad()
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            specific=True)

        self.comb2_left = AvgPoolPad()
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            specific=True)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            specific=True)
        self.comb4_right = MaxPoolPad()

    def forward(self, x_stem_0, x_conv0):
        x_left = self.conv_1x1(x_stem_0)

        # x_relu = self.relu(x_conv0)
        # # path 1
        # x_path1 = self.path_1(x_relu)
        # # path 2
        # x_path2 = self.path_2(x_relu)
        # # x_path2 = self.path_2.pad(x_relu)
        # # x_path2 = x_path2[:, :, 1:, 1:]
        # # x_path2 = self.path_2.avgpool(x_path2)
        # # x_path2 = self.path_2.conv(x_path2)
        # # final path
        # x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.path(x_conv0)

        x_comb0_left = self.comb0_left(x_left)
        x_comb0_right = self.comb0_right(x_right)
        x_comb0 = x_comb0_left + x_comb0_right

        x_comb1_left = self.comb1_left(x_left)
        x_comb1_right = self.comb1_right(x_right)
        x_comb1 = x_comb1_left + x_comb1_right

        x_comb2_left = self.comb2_left(x_left)
        x_comb2_right = self.comb2_right(x_right)
        x_comb2 = x_comb2_left + x_comb2_right

        x_comb3_right = self.comb3_right(x_comb0)
        x_comb3 = x_comb3_right + x_comb1

        x_comb4_left = self.comb4_left(x_comb0)
        x_comb4_right = self.comb4_right(x_left)
        x_comb4 = x_comb4_left + x_comb4_right

        x_out = torch.cat([x_comb1, x_comb2, x_comb3, x_comb4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right):
        super(FirstCell, self).__init__()

        self.conv_1x1 = nas_conv1x1(
            in_channels=in_channels_right,
            out_channels=out_channels_right)

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nasnet_avgpool1x1_s2())
        self.path_1.add_module('conv', conv1x1(
            in_channels=in_channels_left,
            out_channels=out_channels_left))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nasnet_avgpool1x1_s2())
        self.path_2.add_module('conv', conv1x1(
            in_channels=in_channels_left,
            out_channels=out_channels_left))

        self.final_path_bn = nasnet_batch_norm(channels=out_channels_left * 2)

        self.comb0_left = dws_branch_k5_s1_p2(
            in_channels=out_channels_right,
            out_channels=out_channels_right)
        self.comb0_right = dws_branch_k3_s1_p1(
            in_channels=out_channels_right,
            out_channels=out_channels_right)

        self.comb1_left = dws_branch_k5_s1_p2(
            in_channels=out_channels_right,
            out_channels=out_channels_right)
        self.comb1_right = dws_branch_k3_s1_p1(
            in_channels=out_channels_right,
            out_channels=out_channels_right)

        self.comb2_left = nasnet_avgpool3x3_s1()

        self.comb_iter_3_left = nasnet_avgpool3x3_s1()
        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=out_channels_right,
            out_channels=out_channels_right)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_right = self.conv_1x1(x)

        x_comb0_left = self.comb0_left(x_right)
        x_comb0_right = self.comb0_right(x_left)
        x_comb0 = x_comb0_left + x_comb0_right

        x_comb1_left = self.comb1_left(x_left)
        x_comb1_right = self.comb1_right(x_left)
        x_comb1 = x_comb1_left + x_comb1_right

        x_comb2_left = self.comb2_left(x_right)
        x_comb2 = x_comb2_left + x_left

        x_comb3_left = self.comb_iter_3_left(x_left)
        x_comb3_right = self.comb3_right(x_left)
        x_comb3 = x_comb3_left + x_comb3_right

        x_comb4_left = self.comb4_left(x_right)
        x_comb4 = x_comb4_left + x_right

        x_out = torch.cat([x_left, x_comb0, x_comb1, x_comb2, x_comb3, x_comb4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right):
        super(NormalCell, self).__init__()

        self.conv_prev_1x1 = nas_conv1x1(
            in_channels=in_channels_left,
            out_channels=out_channels_left)
        # self.conv_prev_1x1 = nn.Sequential()
        # self.conv_prev_1x1.add_module('relu', nn.ReLU())
        # self.conv_prev_1x1.add_module('conv', conv1x1(
        #     in_channels=in_channels_left,
        #     out_channels=out_channels_left))
        # self.conv_prev_1x1.add_module('bn', nasnet_batch_norm(channels=out_channels_left))

        self.conv_1x1 = nas_conv1x1(
            in_channels=in_channels_right,
            out_channels=out_channels_right)
        # self.conv_1x1 = nn.Sequential()
        # self.conv_1x1.add_module('relu', nn.ReLU())
        # self.conv_1x1.add_module('conv', conv1x1(
        #     in_channels=in_channels_right,
        #     out_channels=out_channels_right))
        # self.conv_1x1.add_module('bn', nasnet_batch_norm(channels=out_channels_right))

        self.comb0_left = dws_branch_k5_s1_p2(
            in_channels=out_channels_right,
            out_channels=out_channels_right)
        self.comb0_right = dws_branch_k3_s1_p1(
            in_channels=out_channels_left,
            out_channels=out_channels_left)

        self.comb1_left = dws_branch_k5_s1_p2(
            in_channels=out_channels_left,
            out_channels=out_channels_left)
        self.comb1_right = dws_branch_k3_s1_p1(
            in_channels=out_channels_left,
            out_channels=out_channels_left)

        self.comb2_left = nasnet_avgpool3x3_s1()

        self.comb_iter_3_left = nasnet_avgpool3x3_s1()
        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=out_channels_right,
            out_channels=out_channels_right)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb0_left = self.comb0_left(x_right)
        x_comb0_right = self.comb0_right(x_left)
        x_comb0 = x_comb0_left + x_comb0_right

        x_comb1_left = self.comb1_left(x_left)
        x_comb1_right = self.comb1_right(x_left)
        x_comb1 = x_comb1_left + x_comb1_right

        x_comb2_left = self.comb2_left(x_right)
        x_comb2 = x_comb2_left + x_left

        x_comb3_left = self.comb_iter_3_left(x_left)
        x_comb3_right = self.comb3_right(x_left)
        x_comb3 = x_comb3_left + x_comb3_right

        x_comb4_left = self.comb4_left(x_right)
        x_comb4 = x_comb4_left + x_right

        x_out = torch.cat([x_left, x_comb0, x_comb1, x_comb2, x_comb3, x_comb4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right):
        super(ReductionCell0, self).__init__()

        self.conv_prev_1x1 = nas_conv1x1(
            in_channels=in_channels_left,
            out_channels=out_channels_left)
        # self.conv_prev_1x1 = nn.Sequential()
        # self.conv_prev_1x1.add_module('relu', nn.ReLU())
        # self.conv_prev_1x1.add_module('conv', conv1x1(
        #     in_channels=in_channels_left,
        #     out_channels=out_channels_left))
        # self.conv_prev_1x1.add_module('bn', nasnet_batch_norm(channels=out_channels_left))

        self.conv_1x1 = nas_conv1x1(
            in_channels=in_channels_right,
            out_channels=out_channels_right)
        # self.conv_1x1 = nn.Sequential()
        # self.conv_1x1.add_module('relu', nn.ReLU())
        # self.conv_1x1.add_module('conv', conv1x1(
        #     in_channels=in_channels_right,
        #     out_channels=out_channels_right))
        # self.conv_1x1.add_module('bn', nasnet_batch_norm(channels=out_channels_right))

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)

        self.comb1_left = MaxPoolPad()
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)

        self.comb2_left = AvgPoolPad()
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)
        self.comb4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb0_left = self.comb0_left(x_right)
        x_comb0_right = self.comb0_right(x_left)
        x_comb0 = x_comb0_left + x_comb0_right

        x_comb1_left = self.comb1_left(x_right)
        x_comb1_right = self.comb1_right(x_left)
        x_comb1 = x_comb1_left + x_comb1_right

        x_comb2_left = self.comb2_left(x_right)
        x_comb2_right = self.comb2_right(x_left)
        x_comb2 = x_comb2_left + x_comb2_right

        x_comb3_right = self.comb3_right(x_comb0)
        x_comb3 = x_comb3_right + x_comb1

        x_comb4_left = self.comb4_left(x_comb0)
        x_comb4_right = self.comb4_right(x_right)
        x_comb4 = x_comb4_left + x_comb4_right

        x_out = torch.cat([x_comb1, x_comb2, x_comb3, x_comb4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right):
        super(ReductionCell1, self).__init__()

        self.conv_prev_1x1 = nas_conv1x1(
            in_channels=in_channels_left,
            out_channels=out_channels_left)
        # self.conv_prev_1x1 = nn.Sequential()
        # self.conv_prev_1x1.add_module('relu', nn.ReLU())
        # self.conv_prev_1x1.add_module('conv', conv1x1(
        #     in_channels=in_channels_left,
        #     out_channels=out_channels_left))
        # self.conv_prev_1x1.add_module('bn', nasnet_batch_norm(channels=out_channels_left))

        self.conv_1x1 = nas_conv1x1(
            in_channels=in_channels_right,
            out_channels=out_channels_right)
        # self.conv_1x1 = nn.Sequential()
        # self.conv_1x1.add_module('relu', nn.ReLU())
        # self.conv_1x1.add_module('conv', conv1x1(
        #     in_channels=in_channels_right,
        #     out_channels=out_channels_right))
        # self.conv_1x1.add_module('bn', nasnet_batch_norm(channels=out_channels_right))

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)

        self.comb1_left = MaxPoolPad()
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)

        self.comb2_left = AvgPoolPad()
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=out_channels_right,
            out_channels=out_channels_right,
            specific=True)
        self.comb4_right =MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb0_left = self.comb0_left(x_right)
        x_comb0_right = self.comb0_right(x_left)
        x_comb0 = x_comb0_left + x_comb0_right

        x_comb1_left = self.comb1_left(x_right)
        x_comb1_right = self.comb1_right(x_left)
        x_comb1 = x_comb1_left + x_comb1_right

        x_comb2_left = self.comb2_left(x_right)
        x_comb2_right = self.comb2_right(x_left)
        x_comb2 = x_comb2_left + x_comb2_right

        x_comb3_right = self.comb3_right(x_comb0)
        x_comb3 = x_comb3_right + x_comb1

        x_comb4_left = self.comb4_left(x_comb0)
        x_comb4_right = self.comb4_right(x_right)
        x_comb4 = x_comb4_left + x_comb4_right

        x_out = torch.cat([x_comb1, x_comb2, x_comb3, x_comb4], 1)
        return x_out


class NASNetInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(NASNetInitBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=0,
            stride=2,
            bias=False)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class NASNetAMobile(nn.Module):
    """NASNetAMobile (4 @ 1056) """

    def __init__(self,
                 init_block_channels,
                 penultimate_filters,
                 in_channels=3,
                 num_classes=1000):
        super(NASNetAMobile, self).__init__()

        #stem_filters = 32
        filters = penultimate_filters // 24
        filters_multiplier = 2

        self.init_block = NASNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels)
        in_channels = init_block_channels

        self.features = DoubleLinkedSequential()

        #in_channels = init_block_channels
        #prev_in_channels = None
        self.features.add_module("cell_stem_0", CellStem0(
            stem_filters=in_channels,
            num_filters=filters // (filters_multiplier ** 2)))
        self.features.add_module("cell_stem_1", CellStem1(
            stem_filters=in_channels,
            num_filters=filters // filters_multiplier))
        #in_channels = init_block_channels
        #prev_in_channels = None

        channels = [[1], [2],
                    [2, 1], [6, 1], [6, 1], [6, 1],
                    [6, 2], [8, 2], [12, 2], [12, 2], [12, 2],
                    [12, 4], [16, 4], [24, 4], [24, 4], [24, 4]]

        self.features.add_module("cell_0", FirstCell(
            in_channels_left=filters,
            out_channels_left=filters//2,
            in_channels_right=2*filters,
            out_channels_right=filters))
        self.features.add_module("cell_1", NormalCell(
            in_channels_left=2*filters,
            out_channels_left=filters,
            in_channels_right=6*filters,
            out_channels_right=filters))
        self.features.add_module("cell_2", NormalCell(
            in_channels_left=6*filters,
            out_channels_left=filters,
            in_channels_right=6*filters,
            out_channels_right=filters))
        self.features.add_module("cell_3", NormalCell(
            in_channels_left=6*filters,
            out_channels_left=filters,
            in_channels_right=6*filters,
            out_channels_right=filters))

        self.features.add_module("reduction_cell_0", ReductionCell0(
            in_channels_left=6*filters,
            out_channels_left=2*filters,
            in_channels_right=6*filters,
            out_channels_right=2*filters))

        self.features.add_module("cell_6", FirstCell(
            in_channels_left=6*filters,
            out_channels_left=filters,
            in_channels_right=8*filters,
            out_channels_right=2*filters))
        self.features.add_module("cell_7", NormalCell(
            in_channels_left=8*filters,
            out_channels_left=2*filters,
            in_channels_right=12*filters,
            out_channels_right=2*filters))
        self.features.add_module("cell_8", NormalCell(
            in_channels_left=12*filters,
            out_channels_left=2*filters,
            in_channels_right=12*filters,
            out_channels_right=2*filters))
        self.features.add_module("cell_9", NormalCell(
            in_channels_left=12*filters,
            out_channels_left=2*filters,
            in_channels_right=12*filters,
            out_channels_right=2*filters))

        self.features.add_module("reduction_cell_1", ReductionCell1(
            in_channels_left=12*filters,
            out_channels_left=4*filters,
            in_channels_right=12*filters,
            out_channels_right=4*filters))

        self.features.add_module("cell_12", FirstCell(
            in_channels_left=12*filters,
            out_channels_left=2*filters,
            in_channels_right=16*filters,
            out_channels_right=4*filters))
        self.features.add_module("cell_13", NormalCell(
            in_channels_left=16*filters,
            out_channels_left=4*filters,
            in_channels_right=24*filters,
            out_channels_right=4*filters))
        self.features.add_module("cell_14", NormalCell(
            in_channels_left=24*filters,
            out_channels_left=4*filters,
            in_channels_right=24*filters,
            out_channels_right=4*filters))
        self.features.add_module("cell_15", NormalCell(
            in_channels_left=24*filters,
            out_channels_left=4*filters,
            in_channels_right=24*filters,
            out_channels_right=4*filters))

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=7,
            stride=1,
            padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(
            in_features=24*filters,
            out_features=num_classes)

    def forward(self, x):
        x = self.init_block(x)
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


def get_nasnet(cell_repeats,
               penultimate_filters,
               pretrained=False,
               **kwargs):

    init_block_channels = 32

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = NASNetAMobile(
        init_block_channels=init_block_channels,
        penultimate_filters=penultimate_filters,
        **kwargs)
    return net


def nasnet_a_mobile(**kwargs):
    return get_nasnet(4, 1056, **kwargs)


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

