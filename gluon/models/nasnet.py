"""
    NASNet-A-Mobile, implemented in Gluon.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.
"""

__all__ = ['NASNet', 'nasnet_a_mobile']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, DualPathSequential


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
                                last_ordinals=0,
                                **kwargs):
    """
    NASNet specific dual path sequential container.
    """
    return DualPathSequential(
        return_two=return_two,
        first_ordinals=first_ordinals,
        last_ordinals=last_ordinals,
        dual_path_scheme=nasnet_dual_path_scheme,
        dual_path_scheme_ordinal=nasnet_dual_path_scheme_ordinal,
        **kwargs)


def nasnet_batch_norm(channels):
    """
    NASNet specific Batch normalization layer.

    Parameters:
    ----------
    channels : int
        Number of channels in input data.
    """
    return nn.BatchNorm(
        momentum=0.1,
        epsilon=0.001,
        in_channels=channels)


def nasnet_maxpool():
    """
    NASNet specific Max pooling layer.
    """
    return nn.MaxPool2D(
        pool_size=3,
        strides=2,
        padding=1)


def nasnet_avgpool1x1_s2():
    """
    NASNet specific 1x1 Average pooling layer with stride 2.
    """
    return nn.AvgPool2D(
        pool_size=1,
        strides=2,
        count_include_pad=False)


def nasnet_avgpool3x3_s1():
    """
    NASNet specific 3x3 Average pooling layer with stride 1.
    """
    return nn.AvgPool2D(
        pool_size=3,
        strides=1,
        padding=1,
        count_include_pad=False)


def nasnet_avgpool3x3_s2():
    """
    NASNet specific 3x3 Average pooling layer with stride 2.
    """
    return nn.AvgPool2D(
        pool_size=3,
        strides=2,
        padding=1,
        count_include_pad=False)


def process_with_padding(x,
                         F,
                         process=(lambda x: x),
                         pad_width=(0, 0, 0, 0, 1, 0, 1, 0)):
    """
    Auxiliary decorator for layer with NASNet specific extra padding.

    Parameters:
    ----------
    x : NDArray
        Input tensor.
    F : module
        Gluon API module.
    process : function, default (lambda x: x)
        a decorated layer
    pad_width : tuple of int, default (0, 0, 0, 0, 1, 0, 1, 0)
        Whether the layer uses a bias vector.

    Returns
    -------
    NDArray
        Resulted tensor.
    """
    x = F.pad(x, mode="constant", pad_width=pad_width, constant_value=0)
    x = process(x)
    x = F.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
    return x


class MaxPoolPad(HybridBlock):
    """
    NASNet specific Max pooling layer with extra padding.
    """
    def __init__(self,
                 **kwargs):
        super(MaxPoolPad, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nasnet_maxpool()

    def hybrid_forward(self, F, x):
        x = process_with_padding(x, F, self.pool)
        return x


class AvgPoolPad(HybridBlock):
    """
    NASNet specific 3x3 Average pooling layer with extra padding.
    """
    def __init__(self,
                 strides=2,
                 padding=1,
                 **kwargs):
        super(AvgPoolPad, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=3,
                strides=strides,
                padding=padding,
                count_include_pad=False)

    def hybrid_forward(self, F, x):
        x = process_with_padding(x, F, self.pool)
        return x


class NasConv(HybridBlock):
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
    strides : int or tuple/list of 2 int
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
                 strides,
                 padding,
                 groups,
                 **kwargs):
        super(NasConv, self).__init__(**kwargs)
        with self.name_scope():
            self.activ = nn.Activation('relu')
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)

    def hybrid_forward(self, F, x):
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
        strides=1,
        padding=0,
        groups=1)


class DwsConv(HybridBlock):
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
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layers use a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 **kwargs):
        super(DwsConv, self).__init__(**kwargs)
        with self.name_scope():
            self.dw_conv = nn.Conv2D(
                channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=in_channels,
                use_bias=use_bias,
                in_channels=in_channels)
            self.pw_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class NasDwsConv(HybridBlock):
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
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    specific : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 specific=False,
                 **kwargs):
        super(NasDwsConv, self).__init__(**kwargs)
        self.specific = specific

        with self.name_scope():
            self.activ = nn.Activation(activation='relu')
            self.conv = DwsConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False)
            self.bn = nasnet_batch_norm(channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.activ(x)
        if self.specific:
            x = process_with_padding(x, F, self.conv)
        else:
            x = self.conv(x)
        x = self.bn(x)
        return x


class DwsBranch(HybridBlock):
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
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    specific : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 specific=False,
                 stem=False,
                 **kwargs):
        super(DwsBranch, self).__init__(**kwargs)
        assert (not stem) or (not specific)
        mid_channels = out_channels if stem else in_channels

        with self.name_scope():
            self.conv1 = NasDwsConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                specific=specific)
            self.conv2 = NasDwsConv(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding)

    def hybrid_forward(self, F, x):
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
        kernel_size=3,
        strides=1,
        padding=1,
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
        kernel_size=5,
        strides=1,
        padding=2,
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
        kernel_size=5,
        strides=2,
        padding=2,
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
        kernel_size=7,
        strides=2,
        padding=3,
        specific=specific,
        stem=stem)


class NasPathBranch(HybridBlock):
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
                 specific=False,
                 **kwargs):
        super(NasPathBranch, self).__init__(**kwargs)
        self.specific = specific

        with self.name_scope():
            self.avgpool = nasnet_avgpool1x1_s2()
            self.conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def hybrid_forward(self, F, x):
        if self.specific:
            x = process_with_padding(x, F, pad_width=(0, 0, 0, 0, 0, 1, 0, 1))
        x = self.avgpool(x)
        x = self.conv(x)
        return x


class NasPathBlock(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(NasPathBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.activ = nn.Activation('relu')
            self.path1 = NasPathBranch(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.path2 = NasPathBranch(
                in_channels=in_channels,
                out_channels=mid_channels,
                specific=True)
            self.bn = nasnet_batch_norm(channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.activ(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = F.concat(x1, x2, dim=1)
        x = self.bn(x)
        return x


class Stem1Unit(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(Stem1Unit, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
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

    def hybrid_forward(self, F, x, _=None):
        x_left = self.conv1x1(x)
        x_right = x

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = F.concat(x1, x2, x3, x4, dim=1)
        return x_out


class Stem2Unit(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(Stem2Unit, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
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

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = F.concat(x1, x2, x3, x4, dim=1)
        return x_out


class FirstUnit(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(FirstUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 6

        with self.name_scope():
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

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.path(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
        x4 = self.comb4_left(x_left) + x_left

        x_out = F.concat(x_right, x0, x1, x2, x3, x4, dim=1)
        return x_out


class NormalUnit(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(NormalUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 6

        with self.name_scope():
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

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + x_right
        x3 = self.comb3_left(x_right) + self.comb3_right(x_right)
        x4 = self.comb4_left(x_left) + x_left

        x_out = F.concat(x_right, x0, x1, x2, x3, x4, dim=1)
        return x_out


class ReductionUnit(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(ReductionUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
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

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv1x1(x)
        x_right = self.conv1x1_prev(x_prev)

        x0 = self.comb0_left(x_left) + self.comb0_right(x_right)
        x1 = self.comb1_left(x_left) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_left) + self.comb2_right(x_right)
        x3 = x1 + self.comb3_right(x0)
        x4 = self.comb4_left(x0) + self.comb4_right(x_left)

        x_out = F.concat(x1, x2, x3, x4, dim=1)
        return x_out


class NASNetInitBlock(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(NASNetInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=3,
                strides=2,
                padding=0,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nasnet_batch_norm(channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class NASNet(HybridBlock):
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
                 classes=1000,
                 **kwargs):
        super(NASNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nasnet_dual_path_sequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=2,
                prefix='')
            self.features.add(NASNetInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels

            out_channels = stem_blocks_channels[0]
            self.features.add(Stem1Unit(
                in_channels=in_channels,
                out_channels=out_channels))
            prev_in_channels = in_channels
            in_channels = out_channels

            out_channels = stem_blocks_channels[1]
            self.features.add(Stem2Unit(
                in_channels=in_channels,
                prev_in_channels=prev_in_channels,
                out_channels=out_channels))
            prev_in_channels = in_channels
            in_channels = out_channels

            for i, channels_per_stage in enumerate(channels):
                stage = nasnet_dual_path_sequential(prefix='stage{}_'.format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        if (j == 0) and (i != 0):
                            unit = ReductionUnit
                        elif ((i == 0) and (j == 0)) or ((i != 0) and (j == 1)):
                            unit = FirstUnit
                        else:
                            unit = NormalUnit
                        stage.add(unit(
                            in_channels=in_channels,
                            prev_in_channels=prev_in_channels,
                            out_channels=out_channels))
                        prev_in_channels = in_channels
                        in_channels = out_channels
                self.features.add(stage)

            self.features.add(nn.Activation('relu'))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dropout(rate=0.5))
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_nasnet(repeat,
               penultimate_filters,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join('~', '.mxnet', 'models'),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(repeat=4, penultimate_filters=1056, model_name="nasnet_a_mobile", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = True

    models = [
        nasnet_a_mobile,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nasnet_a_mobile or weight_count == 5289978)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

