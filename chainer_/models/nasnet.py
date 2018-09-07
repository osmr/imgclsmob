"""
    NASNet-A-Mobile, implemented in Chainer.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.
"""

__all__ = ['NASNet', 'nasnet_a_mobile']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, SimpleSequential, DualPathSequential


def nasnet_dual_path_scheme(block,
                            x,
                            x_prev):
    """
    NASNet specific scheme of dual path response for a block in a DualPathSequential block.

    Parameters:
    ----------
    block : nn.HybridBlock
        A block.
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
    x_next = block(x, x_prev)
    if type(x_next) == tuple:
        x_next, x = x_next
    return x_next, x


def nasnet_dual_path_scheme_ordinal(block,
                                    x,
                                    _):
    """
    NASNet specific scheme of dual path response for an ordinal block with dual inputs/outputs in a DualPathSequential
    block.

    Parameters:
    ----------
    block : nn.HybridBlock
        A block.
    x : Tensor
        Current processed tensor.

    Returns
    -------
    x_next : Tensor
        Next processed tensor.
    x : Tensor
        Current processed tensor.
    """
    return block(x), x


def nasnet_dual_path_sequential(return_two=True,
                                first_ordinals=0,
                                last_ordinals=0):
    """
    NASNet specific dual path sequential container.
    """
    return DualPathSequential(
        return_two=return_two,
        first_ordinals=first_ordinals,
        last_ordinals=last_ordinals,
        dual_path_scheme=nasnet_dual_path_scheme,
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


def nasnet_maxpool():
    """
    NASNet specific Max pooling layer.
    """
    return partial(
        F.max_pooling_2d,
        ksize=3,
        stride=2,
        pad=1,
        cover_all=False)


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
        F.average_pooling_2d,
        ksize=3,
        stride=1,
        pad=1)


def nasnet_avgpool3x3_s2():
    """
    NASNet specific 3x3 Average pooling layer with stride 2.
    """
    return partial(
        F.average_pooling_2d,
        ksize=3,
        stride=2,
        pad=1)


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
    pad_width : tuple of int, default (0, 0, 0, 0, 1, 0, 1, 0)
        Whether the layer uses a bias vector.

    Returns
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted tensor.
    """
    x = F.pad(x, pad_width=pad_width, mode="constant", constant_values=0)
    x = process(x)
    x = x[:, :, 1:, 1:]
    return x


class MaxPoolPad(Chain):
    """
    NASNet specific Max pooling layer with extra padding.
    """
    def __init__(self):
        super(MaxPoolPad, self).__init__()
        with self.init_scope():
            self.pool = nasnet_maxpool()

    def __call__(self, x):
        x = process_with_padding(x, self.pool)
        return x


class AvgPoolPad(Chain):
    """
    NASNet specific 3x3 Average pooling layer with extra padding.

    Parameters:
    ----------
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 stride=2,
                 pad=1):
        super(AvgPoolPad, self).__init__()
        with self.init_scope():
            self.pool = partial(
                F.average_pooling_2d,
                ksize=3,
                stride=stride,
                pad=pad)

    def __call__(self, x):
        x = process_with_padding(x, self.pool)
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
    specific : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 specific=False):
        super(NasDwsConv, self).__init__()
        self.specific = specific

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
        if self.specific:
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
    specific : bool, default False
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
                 specific=False,
                 stem=False):
        super(DwsBranch, self).__init__()
        assert (not stem) or (not specific)
        mid_channels = out_channels if stem else in_channels

        with self.init_scope():
            self.conv1 = NasDwsConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                specific=specific)
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
                        specific=False):
    """
    3x3/1/1 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    specific : bool, default False
        Whether to use extra padding.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=1,
        pad=1,
        specific=specific)


def dws_branch_k5_s1_p2(in_channels,
                        out_channels,
                        specific=False):
    """
    5x5/1/2 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    specific : bool, default False
        Whether to use extra padding.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=5,
        stride=1,
        pad=2,
        specific=specific)


def dws_branch_k5_s2_p2(in_channels,
                        out_channels,
                        specific=False,
                        stem=False):
    """
    5x5/2/2 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    specific : bool, default False
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
        specific=specific,
        stem=stem)


def dws_branch_k7_s2_p3(in_channels,
                        out_channels,
                        specific=False,
                        stem=False):
    """
    7x7/2/3 version of the NASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    specific : bool, default False
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
        specific=specific,
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
    specific : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 specific=False):
        super(NasPathBranch, self).__init__()
        self.specific = specific

        with self.init_scope():
            self.avgpool = nasnet_avgpool1x1_s2()
            self.conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        if self.specific:
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
                specific=True)
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
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels):
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


class ReductionUnit(Chain):
    """
    NASNet Reduction unit (there is only one reduction unit for NASNet-A-Mobile).

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
        super(ReductionUnit, self).__init__()
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
    NASNet (NASNet-A-Mobile) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    stem_blocks_channels : list of 2 int
        Number of output channels for the Stem units.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 stem_blocks_channels,
                 in_channels=3,
                 classes=1000):
        super(NASNet, self).__init__()

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
                    out_channels=out_channels))
                prev_in_channels = in_channels
                in_channels = out_channels

                for i, channels_per_stage in enumerate(channels):
                    stage = nasnet_dual_path_sequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            if (j == 0) and (i != 0):
                                unit = ReductionUnit
                            elif ((i == 0) and (j == 0)) or ((i != 0) and (j == 1)):
                                unit = FirstUnit
                            else:
                                unit = NormalUnit
                            setattr(stage, "unit{}".format(j + 1), unit(
                                in_channels=in_channels,
                                prev_in_channels=prev_in_channels,
                                out_channels=out_channels))
                            prev_in_channels = in_channels
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)

                setattr(self.features, "final_activ", F.relu)
                setattr(self.features, 'final_pool', partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, 'flatten', partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, 'dropout', partial(
                    F.dropout,
                    ratio=0.5))
                setattr(self.output, 'fc', L.Linear(
                    in_size=in_channels,
                    out_size=classes))


    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_nasnet(repeat,
               penultimate_filters,
               model_name=None,
               pretrained=False,
               root=os.path.join('~', '.chainer', 'models'),
               **kwargs):
    """
    Create NASNet (NASNet-A-Mobile) model with specific parameters.

    Parameters:
    ----------
    repeat : int
        NNumber of cell repeats.
    penultimate_filters : int
        Number of filters in the penultimate layer of the network.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    assert (repeat == 4)

    init_block_channels = 32
    stem_blocks_channels = [1, 2]
    channels = [[6, 6, 6, 6],
                [8, 12, 12, 12, 12],
                [16, 24, 24, 24, 24]]
    base_channel_chunk = penultimate_filters // channels[-1][-1]

    stem_blocks_channels = [(ci * base_channel_chunk) for ci in stem_blocks_channels]
    channels = [[(cij * base_channel_chunk) for cij in ci] for ci in channels]

    net = NASNet(
        channels=channels,
        init_block_channels=init_block_channels,
        stem_blocks_channels=stem_blocks_channels,
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


def nasnet_a_mobile(**kwargs):
    """
    NASNet-A-Mobile (NASNet 4x1056) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(repeat=4, penultimate_filters=1056, model_name="nasnet_a_mobile", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = True

    models = [
        nasnet_a_mobile,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nasnet_a_mobile or weight_count == 5289978)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

