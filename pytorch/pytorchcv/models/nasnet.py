"""
    NASNet-A for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.
"""

__all__ = ['NASNet', 'nasnet_4a1056', 'nasnet_6a4032', 'nasnet_dual_path_sequential']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, DualPathSequential


class NasDualPathScheme(object):
    """
    NASNet specific scheme of dual path response for a module in a DualPathSequential module.

    Parameters:
    ----------
    can_skip_input : bool
        Whether can skip input for some modules.
    """
    def __init__(self,
                 can_skip_input):
        super(NasDualPathScheme, self).__init__()
        self.can_skip_input = can_skip_input

    """
    Scheme function.

    Parameters:
    ----------
    module : nn.Module
        A module.
    x : Tensor
        Current processed tensor.
    x_prev : Tensor
        Previous processed tensor.

    Returns
    -------
    x_next : Tensor
        Next processed tensor.
    x : Tensor
        Current processed tensor.
    """
    def __call__(self,
                 module,
                 x,
                 x_prev):
        x_next = module(x, x_prev)
        if type(x_next) == tuple:
            x_next, x = x_next
        if self.can_skip_input and hasattr(module, 'skip_input') and module.skip_input:
            x = x_prev
        return x_next, x


def nasnet_dual_path_scheme_ordinal(module,
                                    x,
                                    _):
    """
    NASNet specific scheme of dual path response for an ordinal module with dual inputs/outputs in a DualPathSequential
    module.

    Parameters:
    ----------
    module : nn.Module
        A module.
    x : Tensor
        Current processed tensor.

    Returns
    -------
    x_next : Tensor
        Next processed tensor.
    x : Tensor
        Current processed tensor.
    """
    return module(x), x


def nasnet_dual_path_sequential(return_two=True,
                                first_ordinals=0,
                                last_ordinals=0,
                                can_skip_input=False):
    """
    NASNet specific dual path sequential container.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first modules with single input/output.
    last_ordinals : int, default 0
        Number of the final modules with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a module.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal module.
    can_skip_input : bool, default False
        Whether can skip input for some modules.
    """
    return DualPathSequential(
        return_two=return_two,
        first_ordinals=first_ordinals,
        last_ordinals=last_ordinals,
        dual_path_scheme=NasDualPathScheme(can_skip_input=can_skip_input),
        dual_path_scheme_ordinal=nasnet_dual_path_scheme_ordinal)


def nasnet_batch_norm(channels):
    """
    NASNet specific Batch normalization layer.

    Parameters:
    ----------
    channels : int
        Number of channels in input data.
    """
    return nn.BatchNorm2d(
        num_features=channels,
        eps=0.001,
        momentum=0.1,
        affine=True)


def nasnet_avgpool1x1_s2():
    """
    NASNet specific 1x1 Average pooling layer with stride 2.
    """
    return nn.AvgPool2d(
        kernel_size=1,
        stride=2,
        count_include_pad=False)


def nasnet_avgpool3x3_s1():
    """
    NASNet specific 3x3 Average pooling layer with stride 1.
    """
    return nn.AvgPool2d(
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False)


def nasnet_avgpool3x3_s2():
    """
    NASNet specific 3x3 Average pooling layer with stride 2.
    """
    return nn.AvgPool2d(
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False)


class NasMaxPoolBlock(nn.Module):
    """
    NASNet specific Max pooling layer with extra padding.

    Parameters:
    ----------
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 extra_padding=False):
        super(NasMaxPoolBlock, self).__init__()
        self.extra_padding = extra_padding

        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        if self.extra_padding:
            x = self.pad(x)
        x = self.pool(x)
        if self.extra_padding:
            x = x[:, :, 1:, 1:].contiguous()
        return x


class NasAvgPoolBlock(nn.Module):
    """
    NASNet specific 3x3 Average pooling layer with extra padding.

    Parameters:
    ----------
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 extra_padding=False):
        super(NasAvgPoolBlock, self).__init__()
        self.extra_padding = extra_padding

        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=False)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        if self.extra_padding:
            x = self.pad(x)
        x = self.pool(x)
        if self.extra_padding:
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
    """
    Standard depthwise separable convolution block.

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
    bias : bool, default False
        Whether the layers use a bias vector.
    """
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
    NASNet specific depthwise separable convolution block.

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
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 extra_padding=False):
        super(NasDwsConv, self).__init__()
        self.extra_padding = extra_padding

        self.activ = nn.ReLU()
        self.conv = DwsConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nasnet_batch_norm(channels=out_channels)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        x = self.activ(x)
        if self.extra_padding:
            x = self.pad(x)
        x = self.conv(x)
        if self.extra_padding:
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn(x)
        return x


class DwsBranch(nn.Module):
    """
    NASNet specific block with depthwise separable convolution layers.

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
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 extra_padding=False,
                 stem=False):
        super(DwsBranch, self).__init__()
        assert (not stem) or (not extra_padding)
        mid_channels = out_channels if stem else in_channels

        self.conv1 = NasDwsConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            extra_padding=extra_padding)
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
                        extra_padding=False):
    """
    3x3/1/1 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        extra_padding=extra_padding)


def dws_branch_k5_s1_p2(in_channels,
                        out_channels,
                        extra_padding=False):
    """
    5x5/1/2 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=1,
        padding=2,
        extra_padding=extra_padding)


def dws_branch_k5_s2_p2(in_channels,
                        out_channels,
                        extra_padding=False,
                        stem=False):
    """
    5x5/2/2 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=2,
        padding=2,
        extra_padding=extra_padding,
        stem=stem)


def dws_branch_k7_s2_p3(in_channels,
                        out_channels,
                        extra_padding=False,
                        stem=False):
    """
    7x7/2/3 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=2,
        padding=3,
        extra_padding=extra_padding,
        stem=stem)


class NasPathBranch(nn.Module):
    """
    NASNet specific `path` branch (auxiliary block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 extra_padding=False):
        super(NasPathBranch, self).__init__()
        self.extra_padding = extra_padding

        self.avgpool = nasnet_avgpool1x1_s2()
        self.conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))

    def forward(self, x):
        if self.extra_padding:
            x = self.pad(x)
            x = x[:, :, 1:, 1:].contiguous()
        x = self.avgpool(x)
        x = self.conv(x)
        return x


class NasPathBlock(nn.Module):
    """
    NASNet specific `path` block.

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
            extra_padding=True)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.activ(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        return x


class Stem1Unit(nn.Module):
    """
    NASNet Stem1 unit.

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

        self.comb1_left = NasMaxPoolBlock(extra_padding=False)
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
        self.comb4_right = NasMaxPoolBlock(extra_padding=False)

    def forward(self, x, _=None):
        x_left = self.conv1x1(x)
        x_right = x

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        return x_out


class Stem2Unit(nn.Module):
    """
    NASNet Stem2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    extra_padding : bool
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 extra_padding):
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
            extra_padding=extra_padding)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)

        self.comb1_left = NasMaxPoolBlock(extra_padding=extra_padding)
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)

        self.comb2_left = NasAvgPoolBlock(extra_padding=extra_padding)
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)
        self.comb4_right = NasMaxPoolBlock(extra_padding=extra_padding)

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        return x_out


class FirstUnit(nn.Module):
    """
    NASNet First unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    """
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
        return x_out


class NormalUnit(nn.Module):
    """
    NASNet Normal unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    """
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
        return x_out


class ReductionBaseUnit(nn.Module):
    """
    NASNet Reduction base unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    extra_padding : bool, default True
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 extra_padding=True):
        super(ReductionBaseUnit, self).__init__()
        self.skip_input = True
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
            extra_padding=extra_padding)
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)

        self.comb1_left = NasMaxPoolBlock(extra_padding=extra_padding)
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)

        self.comb2_left = NasAvgPoolBlock(extra_padding=extra_padding)
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)

        self.comb3_right = nasnet_avgpool3x3_s1()

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding)
        self.comb4_right = NasMaxPoolBlock(extra_padding=extra_padding)

    def forward(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        return x_out


class Reduction1Unit(ReductionBaseUnit):
    """
    NASNet Reduction1 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels):
        super(Reduction1Unit, self).__init__(
            in_channels=in_channels,
            prev_in_channels=prev_in_channels,
            out_channels=out_channels,
            extra_padding=True)


class Reduction2Unit(ReductionBaseUnit):
    """
    NASNet Reduction2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    extra_padding : bool
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 extra_padding):
        super(Reduction2Unit, self).__init__(
            in_channels=in_channels,
            prev_in_channels=prev_in_channels,
            out_channels=out_channels,
            extra_padding=extra_padding)


class NASNetInitBlock(nn.Module):
    """
    NASNet specific initial block.

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
    """
    NASNet-A model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    stem_blocks_channels : list of 2 int
        Number of output channels for the Stem units.
    final_pool_size : int
        Size of the pooling windows for final pool.
    extra_padding : bool
        Whether to use extra padding.
    skip_reduction_layer_input : bool
        Whether to skip the reduction layers when calculating the previous layer to connect to.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 stem_blocks_channels,
                 final_pool_size,
                 extra_padding,
                 skip_reduction_layer_input,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(NASNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        reduction_units = [Reduction1Unit, Reduction2Unit]

        self.features = nasnet_dual_path_sequential(
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
            out_channels=out_channels,
            extra_padding=extra_padding))
        prev_in_channels = in_channels
        in_channels = out_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nasnet_dual_path_sequential(can_skip_input=skip_reduction_layer_input)
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                elif ((i == 0) and (j == 0)) or ((i != 0) and (j == 1)):
                    unit = FirstUnit
                else:
                    unit = NormalUnit
                if unit == Reduction2Unit:
                    stage.add_module("unit{}".format(j + 1), Reduction2Unit(
                        in_channels=in_channels,
                        prev_in_channels=prev_in_channels,
                        out_channels=out_channels,
                        extra_padding=extra_padding))
                else:
                    stage.add_module("unit{}".format(j + 1), unit(
                        in_channels=in_channels,
                        prev_in_channels=prev_in_channels,
                        out_channels=out_channels))
                prev_in_channels = in_channels
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("activ", nn.ReLU())
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=final_pool_size,
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
               init_block_channels,
               final_pool_size,
               extra_padding,
               skip_reduction_layer_input,
               in_size,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create NASNet-A model with specific parameters.

    Parameters:
    ----------
    repeat : int
        NNumber of cell repeats.
    penultimate_filters : int
        Number of filters in the penultimate layer of the network.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_pool_size : int
        Size of the pooling windows for final pool.
    extra_padding : bool
        Whether to use extra padding.
    skip_reduction_layer_input : bool
        Whether to skip the reduction layers when calculating the previous layer to connect to.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    stem_blocks_channels = [1, 2]
    reduct_channels = [[], [8], [16]]
    norm_channels = [6, 12, 24]
    channels = [rci + [nci] * repeat for rci, nci in zip(reduct_channels, norm_channels)]

    base_channel_chunk = penultimate_filters // channels[-1][-1]

    stem_blocks_channels = [(ci * base_channel_chunk) for ci in stem_blocks_channels]
    channels = [[(cij * base_channel_chunk) for cij in ci] for ci in channels]

    net = NASNet(
        channels=channels,
        init_block_channels=init_block_channels,
        stem_blocks_channels=stem_blocks_channels,
        final_pool_size=final_pool_size,
        extra_padding=extra_padding,
        skip_reduction_layer_input=skip_reduction_layer_input,
        in_size=in_size,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def nasnet_4a1056(**kwargs):
    """
    NASNet-A 4@1056 (NASNet-A-Mobile) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(
        repeat=4,
        penultimate_filters=1056,
        init_block_channels=32,
        final_pool_size=7,
        extra_padding=True,
        skip_reduction_layer_input=False,
        in_size=(224, 224),
        model_name="nasnet_4a1056",
        **kwargs)


def nasnet_6a4032(**kwargs):
    """
    NASNet-A 6@4032 (NASNet-A-Large) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(
        repeat=6,
        penultimate_filters=4032,
        init_block_channels=96,
        final_pool_size=11,
        extra_padding=False,
        skip_reduction_layer_input=True,
        in_size=(331, 331),
        model_name="nasnet_6a4032",
        **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        nasnet_4a1056,
        nasnet_6a4032,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nasnet_4a1056 or weight_count == 5289978)
        assert (model != nasnet_6a4032 or weight_count == 88753150)

        x = torch.randn(1, 3, net.in_size[0], net.in_size[1])
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
