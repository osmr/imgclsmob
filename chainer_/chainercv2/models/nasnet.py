"""
    NASNet-A for ImageNet-1K, implemented in Chainer.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.
"""

__all__ = ['NASNet', 'nasnet_4a1056', 'nasnet_6a4032', 'nasnet_dual_path_sequential']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, SimpleSequential, DualPathSequential


class NasDualPathScheme(object):
    """
    NASNet specific scheme of dual path response for a module in a DualPathSequential module.

    Parameters:
    ----------
    can_skip_input : bool
        Whether can skip input for some blocks.
    """

    def __init__(self,
                 can_skip_input):
        super(NasDualPathScheme, self).__init__()
        self.can_skip_input = can_skip_input

    """
    Scheme function.

    Parameters:
    ----------
    block : Chain
        A block.
    x : Tensor
        Current processed tensor.
    x_prev : Tensor
        Previous processed tensor.

    Returns:
    -------
    x_next : Tensor
        Next processed tensor.
    x : Tensor
        Current processed tensor.
    """

    def __call__(self,
                 block,
                 x,
                 x_prev):
        x_next = block(x, x_prev)
        if type(x_next) == tuple:
            x_next, x = x_next
        if self.can_skip_input and hasattr(block, 'skip_input') and block.skip_input:
            x = x_prev
        return x_next, x


def nasnet_dual_path_scheme_ordinal(block,
                                    x,
                                    _):
    """
    NASNet specific scheme of dual path response for an ordinal block with dual inputs/outputs in a DualPathSequential
    block.

    Parameters:
    ----------
    block : Chain
        A block.
    x : Tensor
        Current processed tensor.

    Returns:
    -------
    x_next : Tensor
        Next processed tensor.
    x : Tensor
        Current processed tensor.
    """
    return block(x), x


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
        Number of the first blocks with single input/output.
    last_ordinals : int, default 0
        Number of the final blocks with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a block.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal block.
    can_skip_input : bool, default False
        Whether can skip input for some blocks.
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
    return L.BatchNormalization(
        size=channels,
        decay=0.1,
        eps=0.001)


def nasnet_avgpool1x1_s2():
    """
    NASNet specific 1x1 Average pooling layer with stride 2.
    """
    return partial(
        F.average_pooling_2d,
        ksize=1,
        stride=2)


def nasnet_avgpool3x3_s1():
    """
    NASNet specific 3x3 Average pooling layer with stride 1.
    """
    return partial(
        F.average_pooling_nd,
        ksize=3,
        stride=1,
        pad=1,
        pad_value=None)


def nasnet_avgpool3x3_s2():
    """
    NASNet specific 3x3 Average pooling layer with stride 2.
    """
    return partial(
        F.average_pooling_nd,
        ksize=3,
        stride=2,
        pad=1,
        pad_value=None)


def process_with_padding(x,
                         process=(lambda x: x),
                         pad_width=((0, 0), (0, 0), (1, 0), (1, 0))):
    """
    Auxiliary decorator for layer with NASNet specific extra padding.

    Parameters:
    ----------
    x : chainer.Variable or numpy.ndarray or cupy.ndarray
        Input tensor.
    process : function, default (lambda x: x)
        a decorated layer
    pad_width : tuple of tuple of int, default ((0, 0), (0, 0), (1, 0), (1, 0))
        Whether the layer uses a bias vector.

    Returns:
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted tensor.
    """
    x = F.pad(x, pad_width=pad_width, mode="constant", constant_values=0)
    x = process(x)
    x = x[:, :, 1:, 1:]
    return x


class NasMaxPoolBlock(Chain):
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

        with self.init_scope():
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        if self.extra_padding:
            x = process_with_padding(x, self.pool)
        else:
            x = self.pool(x)
        return x


class NasAvgPoolBlock(Chain):
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

        with self.init_scope():
            self.pool = partial(
                F.average_pooling_nd,
                ksize=3,
                stride=2,
                pad=1,
                pad_value=None)

    def __call__(self, x):
        if self.extra_padding:
            x = process_with_padding(x, self.pool)
        else:
            x = self.pool(x)
        return x


class NasConv(Chain):
    """
    NASNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 groups):
        super(NasConv, self).__init__()
        with self.init_scope():
            self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True,
                groups=groups)
            self.bn = nasnet_batch_norm(channels=out_channels)

    def __call__(self, x):
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
        ksize=1,
        stride=1,
        pad=0,
        groups=1)


class DwsConv(Chain):
    """
    Standard depthwise separable convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layers use a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 use_bias=False):
        super(DwsConv, self).__init__()
        with self.init_scope():
            self.dw_conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=in_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                groups=in_channels)
            self.pw_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias)

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class NasDwsConv(Chain):
    """
    NASNet specific depthwise separable convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 extra_padding=False):
        super(NasDwsConv, self).__init__()
        self.extra_padding = extra_padding

        with self.init_scope():
            self.activ = F.relu
            self.conv = DwsConv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                use_bias=False)
            self.bn = nasnet_batch_norm(channels=out_channels)

    def __call__(self, x):
        x = self.activ(x)
        if self.extra_padding:
            x = process_with_padding(x, self.conv)
        else:
            x = self.conv(x)
        x = self.bn(x)
        return x


class DwsBranch(Chain):
    """
    NASNet specific block with depthwise separable convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    ding : int or tuple/list of 2 int
        Padding value for convolution layer.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 extra_padding=False,
                 stem=False):
        super(DwsBranch, self).__init__()
        assert (not stem) or (not extra_padding)
        mid_channels = out_channels if stem else in_channels

        with self.init_scope():
            self.conv1 = NasDwsConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                extra_padding=extra_padding)
            self.conv2 = NasDwsConv(
                in_channels=mid_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=1,
                pad=pad)

    def __call__(self, x):
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
        ksize=3,
        stride=1,
        pad=1,
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
        ksize=5,
        stride=1,
        pad=2,
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
        ksize=5,
        stride=2,
        pad=2,
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
        ksize=7,
        stride=2,
        pad=3,
        extra_padding=extra_padding,
        stem=stem)


class NasPathBranch(Chain):
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

        with self.init_scope():
            self.avgpool = nasnet_avgpool1x1_s2()
            self.conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        if self.extra_padding:
            x = process_with_padding(x, pad_width=((0, 0), (0, 0), (0, 1), (0, 1)))
        x = self.avgpool(x)
        x = self.conv(x)
        return x


class NasPathBlock(Chain):
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

        with self.init_scope():
            self.activ = F.relu
            self.path1 = NasPathBranch(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.path2 = NasPathBranch(
                in_channels=in_channels,
                out_channels=mid_channels,
                extra_padding=True)
            self.bn = nasnet_batch_norm(channels=out_channels)

    def __call__(self, x):
        x = self.activ(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = F.concat((x1, x2), axis=1)
        x = self.bn(x)
        return x


class Stem1Unit(Chain):
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

        with self.init_scope():
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

    def __call__(self, x, _=None):
        x_left = self.conv1x1(x)
        x_right = x

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = F.concat((x1, x2, x3, x4), axis=1)
        return x_out


class Stem2Unit(Chain):
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

        with self.init_scope():
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

    def __call__(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = F.concat((x1, x2, x3, x4), axis=1)
        return x_out


class FirstUnit(Chain):
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

        with self.init_scope():
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

    def __call__(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
        x4 = self.comb4_left(x_left) + x_left

        x_out = F.concat((x_right, x0, x1, x2, x3, x4), axis=1)
        return x_out


class NormalUnit(Chain):
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

        with self.init_scope():
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

    def __call__(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
        x4 = self.comb4_left(x_left) + x_left

        x_out = F.concat((x_right, x0, x1, x2, x3, x4), axis=1)
        return x_out


class ReductionBaseUnit(Chain):
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

        with self.init_scope():
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

    def __call__(self, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = F.concat((x1, x2, x3, x4), axis=1)
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


class NASNetInitBlock(Chain):
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
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=3,
                stride=2,
                pad=0,
                nobias=True)
            self.bn = nasnet_batch_norm(channels=out_channels)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class NASNet(Chain):
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
    classes : int, default 1000
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
                 classes=1000):
        super(NASNet, self).__init__()
        self.in_size = in_size
        self.classes = classes
        reduction_units = [Reduction1Unit, Reduction2Unit]

        with self.init_scope():
            self.features = nasnet_dual_path_sequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=2)
            with self.features.init_scope():
                setattr(self.features, "init_block", NASNetInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels

                out_channels = stem_blocks_channels[0]
                setattr(self.features, "stem1_unit", Stem1Unit(
                    in_channels=in_channels,
                    out_channels=out_channels))
                prev_in_channels = in_channels
                in_channels = out_channels

                out_channels = stem_blocks_channels[1]
                setattr(self.features, "stem2_unit", Stem2Unit(
                    in_channels=in_channels,
                    prev_in_channels=prev_in_channels,
                    out_channels=out_channels,
                    extra_padding=extra_padding))
                prev_in_channels = in_channels
                in_channels = out_channels

                for i, channels_per_stage in enumerate(channels):
                    stage = nasnet_dual_path_sequential(can_skip_input=skip_reduction_layer_input)
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            if (j == 0) and (i != 0):
                                unit = reduction_units[i - 1]
                            elif ((i == 0) and (j == 0)) or ((i != 0) and (j == 1)):
                                unit = FirstUnit
                            else:
                                unit = NormalUnit
                            if unit == Reduction2Unit:
                                setattr(stage, "unit{}".format(j + 1), Reduction2Unit(
                                    in_channels=in_channels,
                                    prev_in_channels=prev_in_channels,
                                    out_channels=out_channels,
                                    extra_padding=extra_padding))
                            else:
                                setattr(stage, "unit{}".format(j + 1), unit(
                                    in_channels=in_channels,
                                    prev_in_channels=prev_in_channels,
                                    out_channels=out_channels))
                            prev_in_channels = in_channels
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)

                setattr(self.features, "final_activ", F.relu)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=final_pool_size,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "dropout", partial(
                    F.dropout,
                    ratio=0.5))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
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
               root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def nasnet_4a1056(**kwargs):
    """
    NASNet-A 4@1056 (NASNet-A-Mobile) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        nasnet_4a1056,
        nasnet_6a4032,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nasnet_4a1056 or weight_count == 5289978)
        assert (model != nasnet_6a4032 or weight_count == 88753150)

        x = np.zeros((1, 3, net.in_size[0], net.in_size[1]), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
