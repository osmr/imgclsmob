"""
    NASNet-A for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.
"""

__all__ = ['NASNet', 'nasnet_4a1056', 'nasnet_6a4032', 'nasnet_dual_path_sequential']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, AvgPool2d, BatchNorm, Conv2d, conv1x1, DualPathSequential, SimpleSequential, flatten,\
    is_channels_first, get_channel_axis


class NasDualPathScheme(object):
    """
    NASNet specific scheme of dual path response for a block in a DualPathSequential module.

    Parameters:
    ----------
    can_skip_input : bool
        Whether can skip input for some blocks.
    """
    def __init__(self,
                 can_skip_input):
        super(NasDualPathScheme, self).__init__()
        self.can_skip_input = can_skip_input

    def __call__(self,
                 block,
                 x,
                 x_prev,
                 training):
        """
        Scheme function.

        Parameters:
        ----------
        block : nn.HybridBlock
            A block.
        x : Tensor
            Current processed tensor.
        x_prev : Tensor
            Previous processed tensor.
        training : bool or None
            Whether to work in training mode or in inference mode.

        Returns:
        -------
        x_next : Tensor
            Next processed tensor.
        x : Tensor
            Current processed tensor.
        """
        x_next = block(x, x_prev, training=training)
        if type(x_next) == tuple:
            x_next, x = x_next
        if self.can_skip_input and hasattr(block, 'skip_input') and block.skip_input:
            x = x_prev
        return x_next, x


def nasnet_dual_path_scheme_ordinal(block,
                                    x,
                                    _,
                                    training):
    """
    NASNet specific scheme of dual path response for an ordinal block with dual inputs/outputs in a DualPathSequential
    block.

    Parameters:
    ----------
    block : nn.HybridBlock
        A block.
    x : Tensor
        Current processed tensor.
    training : bool or None
        Whether to work in training mode or in inference mode.

    Returns:
    -------
    x_next : Tensor
        Next processed tensor.
    x : Tensor
        Current processed tensor.
    """
    return block(x, training=training), x


def nasnet_dual_path_sequential(return_two=True,
                                first_ordinals=0,
                                last_ordinals=0,
                                can_skip_input=False,
                                **kwargs):
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
        dual_path_scheme_ordinal=nasnet_dual_path_scheme_ordinal,
        **kwargs)


def nasnet_batch_norm(channels,
                      data_format="channels_last",
                      **kwargs):
    """
    NASNet specific Batch normalization layer.

    Parameters:
    ----------
    channels : int
        Number of channels in input data.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    assert (channels is not None)
    return BatchNorm(
        momentum=0.1,
        epsilon=0.001,
        data_format=data_format,
        **kwargs)


def nasnet_avgpool1x1_s2(data_format="channels_last",
                         **kwargs):
    """
    NASNet specific 1x1 Average pooling layer with stride 2.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return AvgPool2d(
        pool_size=1,
        strides=2,
        # count_include_pad=False,
        data_format=data_format,
        **kwargs)


def nasnet_avgpool3x3_s1(data_format="channels_last",
                         **kwargs):
    """
    NASNet specific 3x3 Average pooling layer with stride 1.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return AvgPool2d(
        pool_size=3,
        strides=1,
        padding=1,
        # count_include_pad=False,
        data_format=data_format,
        **kwargs)


def nasnet_avgpool3x3_s2(data_format="channels_last",
                         **kwargs):
    """
    NASNet specific 3x3 Average pooling layer with stride 2.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return AvgPool2d(
        pool_size=3,
        strides=2,
        padding=1,
        # count_include_pad=False,
        data_format=data_format,
        **kwargs)


class NasMaxPoolBlock(nn.Layer):
    """
    NASNet specific Max pooling layer with extra padding.

    Parameters:
    ----------
    extra_padding : bool, default False
        Whether to use extra padding.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 extra_padding=False,
                 data_format="channels_last",
                 **kwargs):
        super(NasMaxPoolBlock, self).__init__(**kwargs)
        self.extra_padding = extra_padding
        self.data_format = data_format

        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")
        if self.extra_padding:
            self.pad = nn.ZeroPadding2D(
                padding=((1, 0), (1, 0)),
                data_format=data_format)

    def call(self, x, training=None):
        if self.extra_padding:
            x = self.pad(x)
        x = self.pool(x)
        if self.extra_padding:
            if is_channels_first(self.data_format):
                x = x[:, :, 1:, 1:]
            else:
                x = x[:, 1:, 1:, :]
        return x


class NasAvgPoolBlock(nn.Layer):
    """
    NASNet specific 3x3 Average pooling layer with extra padding.

    Parameters:
    ----------
    extra_padding : bool, default False
        Whether to use extra padding.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 extra_padding=False,
                 data_format="channels_last",
                 **kwargs):
        super(NasAvgPoolBlock, self).__init__(**kwargs)
        self.extra_padding = extra_padding
        self.data_format = data_format

        self.pool = AvgPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            # count_include_pad=False,
            data_format=data_format,
            name="pool")
        if self.extra_padding:
            self.pad = nn.ZeroPadding2D(
                padding=((1, 0), (1, 0)),
                data_format=data_format)

    def call(self, x, training=None):
        if self.extra_padding:
            x = self.pad(x)
        x = self.pool(x)
        if self.extra_padding:
            if is_channels_first(self.data_format):
                x = x[:, :, 1:, 1:]
            else:
                x = x[:, 1:, 1:, :]
        return x


class NasConv(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups,
                 data_format="channels_last",
                 **kwargs):
        super(NasConv, self).__init__(**kwargs)
        self.activ = nn.ReLU()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = nasnet_batch_norm(
            channels=out_channels,
            data_format=data_format,
            name="bn")

    def call(self, x, training=None):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


def nas_conv1x1(in_channels,
                out_channels,
                data_format="channels_last",
                **kwargs):
    """
    1x1 version of the NASNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return NasConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        groups=1,
        data_format=data_format,
        **kwargs)


class DwsConv(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 data_format="channels_last",
                 **kwargs):
        super(DwsConv, self).__init__(**kwargs)
        self.dw_conv = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=in_channels,
            use_bias=use_bias,
            data_format=data_format,
            name="dw_conv")
        self.pw_conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            data_format=data_format,
            name="pw_conv")

    def call(self, x, training=None):
        x = self.dw_conv(x, training=training)
        x = self.pw_conv(x, training=training)
        return x


class NasDwsConv(nn.Layer):
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
    extra_padding : bool, default False
        Whether to use extra padding.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 extra_padding=False,
                 data_format="channels_last",
                 **kwargs):
        super(NasDwsConv, self).__init__(**kwargs)
        self.extra_padding = extra_padding
        self.data_format = data_format

        self.activ = nn.ReLU()
        self.conv = DwsConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = nasnet_batch_norm(
            channels=out_channels,
            data_format=data_format,
            name="bn")
        if self.extra_padding:
            self.pad = nn.ZeroPadding2D(
                padding=((1, 0), (1, 0)),
                data_format=data_format)

    def call(self, x, training=None):
        x = self.activ(x)
        if self.extra_padding:
            x = self.pad(x)
        x = self.conv(x, training=training)
        if self.extra_padding:
            if is_channels_first(self.data_format):
                x = x[:, :, 1:, 1:]
            else:
                x = x[:, 1:, 1:, :]
        x = self.bn(x, training=training)
        return x


class DwsBranch(nn.Layer):
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
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 extra_padding=False,
                 stem=False,
                 data_format="channels_last",
                 **kwargs):
        super(DwsBranch, self).__init__(**kwargs)
        assert (not stem) or (not extra_padding)
        mid_channels = out_channels if stem else in_channels

        self.conv1 = NasDwsConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            extra_padding=extra_padding,
            data_format=data_format,
            name="conv1")
        self.conv2 = NasDwsConv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


def dws_branch_k3_s1_p1(in_channels,
                        out_channels,
                        extra_padding=False,
                        data_format="channels_last",
                        **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1,
        extra_padding=extra_padding,
        data_format=data_format,
        **kwargs)


def dws_branch_k5_s1_p2(in_channels,
                        out_channels,
                        extra_padding=False,
                        data_format="channels_last",
                        **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=1,
        padding=2,
        extra_padding=extra_padding,
        data_format=data_format,
        **kwargs)


def dws_branch_k5_s2_p2(in_channels,
                        out_channels,
                        extra_padding=False,
                        stem=False,
                        data_format="channels_last",
                        **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=2,
        padding=2,
        extra_padding=extra_padding,
        stem=stem,
        data_format=data_format,
        **kwargs)


def dws_branch_k7_s2_p3(in_channels,
                        out_channels,
                        extra_padding=False,
                        stem=False,
                        data_format="channels_last",
                        **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=2,
        padding=3,
        extra_padding=extra_padding,
        stem=stem,
        data_format=data_format,
        **kwargs)


class NasPathBranch(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 extra_padding=False,
                 data_format="channels_last",
                 **kwargs):
        super(NasPathBranch, self).__init__(**kwargs)
        self.extra_padding = extra_padding
        self.data_format = data_format

        self.avgpool = nasnet_avgpool1x1_s2(
            data_format=data_format,
            name="")
        self.conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="")
        if self.extra_padding:
            self.pad = nn.ZeroPadding2D(
                padding=((0, 1), (0, 1)),
                data_format=data_format)

    def call(self, x, training=None):
        if self.extra_padding:
            x = self.pad(x)
            if is_channels_first(self.data_format):
                x = x[:, :, 1:, 1:]
            else:
                x = x[:, 1:, 1:, :]
        x = self.avgpool(x)
        x = self.conv(x, training=training)
        return x


class NasPathBlock(nn.Layer):
    """
    NASNet specific `path` block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(NasPathBlock, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = out_channels // 2

        self.activ = nn.ReLU()
        self.path1 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="path1")
        self.path2 = NasPathBranch(
            in_channels=in_channels,
            out_channels=mid_channels,
            extra_padding=True,
            data_format=data_format,
            name="path2")
        self.bn = nasnet_batch_norm(
            channels=out_channels,
            data_format=data_format,
            name="bn")

    def call(self, x, training=None):
        x = self.activ(x)
        x1 = self.path1(x, training=training)
        x2 = self.path2(x, training=training)
        x = tf.concat([x1, x2], axis=get_channel_axis(self.data_format))
        x = self.bn(x, training=training)
        return x


class Stem1Unit(nn.Layer):
    """
    NASNet Stem1 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(Stem1Unit, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = out_channels // 4

        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1x1")

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb0_left")
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True,
            data_format=data_format,
            name="comb0_right")

        self.comb1_left = NasMaxPoolBlock(
            extra_padding=False,
            data_format=data_format,
            name="comb1_left")
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True,
            data_format=data_format,
            name="comb1_right")

        self.comb2_left = nasnet_avgpool3x3_s2(
            data_format=data_format,
            name="comb2_left")
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True,
            data_format=data_format,
            name="comb2_right")

        self.comb3_right = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb3_right")

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb4_left")
        self.comb4_right = NasMaxPoolBlock(
            extra_padding=False,
            data_format=data_format,
            name="comb4_right")

    def call(self, x, _=None, training=None):
        x_left = self.conv1x1(x, training=training)
        x_right = x

        x0 = self.comb0_left(x_left, training=training) + self.comb0_right(x_right, training=training)
        x1 = self.comb1_left(x_left, training=training) + self.comb1_right(x_right, training=training)
        x2 = self.comb2_left(x_left, training=training) + self.comb2_right(x_right, training=training)
        x3 = x1 + self.comb3_right(x0, training=training)
        x4 = self.comb4_left(x0, training=training) + self.comb4_right(x_left, training=training)

        x_out = tf.concat([x1, x2, x3, x4], axis=get_channel_axis(self.data_format))
        return x_out


class Stem2Unit(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 extra_padding,
                 data_format="channels_last",
                 **kwargs):
        super(Stem2Unit, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = out_channels // 4

        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1x1")
        self.path = NasPathBlock(
            in_channels=prev_in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="path")

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb0_left")
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb0_right")

        self.comb1_left = NasMaxPoolBlock(
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb1_left")
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb1_right")

        self.comb2_left = NasAvgPoolBlock(
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb2_left")
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb2_right")

        self.comb3_right = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb3_right")

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb4_left")
        self.comb4_right = NasMaxPoolBlock(
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb4_right")

    def call(self, x, x_prev, training=None):
        x_left = self.conv1x1(x, training=training)
        x_right = self.path(x_prev, training=training)

        x0 = self.comb0_left(x_left, training=training) + self.comb0_right(x_right, training=training)
        x1 = self.comb1_left(x_left, training=training) + self.comb1_right(x_right, training=training)
        x2 = self.comb2_left(x_left, training=training) + self.comb2_right(x_right, training=training)
        x3 = x1 + self.comb3_right(x0, training=training)
        x4 = self.comb4_left(x0, training=training) + self.comb4_right(x_left, training=training)

        x_out = tf.concat([x1, x2, x3, x4], axis=get_channel_axis(self.data_format))
        return x_out


class FirstUnit(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(FirstUnit, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = out_channels // 6

        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1x1")

        self.path = NasPathBlock(
            in_channels=prev_in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="path")

        self.comb0_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb0_left")
        self.comb0_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb0_right")

        self.comb1_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb1_left")
        self.comb1_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb1_right")

        self.comb2_left = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb2_left")

        self.comb3_left = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb3_left")
        self.comb3_right = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb3_right")

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb4_left")

    def call(self, x, x_prev, training=None):
        x_left = self.conv1x1(x, training=training)
        x_right = self.path(x_prev, training=training)

        x0 = self.comb0_left(x_left, training=training) + self.comb0_right(x_right, training=training)
        x1 = self.comb1_left(x_right, training=training) + self.comb1_right(x_right, training=training)
        x2 = self.comb2_left(x_left, training=training) + x_right
        x3 = self.comb3_left(x_right, training=training) + self.comb3_right(x_right, training=training)
        x4 = self.comb4_left(x_left, training=training) + x_left

        x_out = tf.concat([x_right, x0, x1, x2, x3, x4], axis=get_channel_axis(self.data_format))
        return x_out


class NormalUnit(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(NormalUnit, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = out_channels // 6

        self.conv1x1_prev = nas_conv1x1(
            in_channels=prev_in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1x1_prev")
        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1x1")

        self.comb0_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb0_left")
        self.comb0_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb0_right")

        self.comb1_left = dws_branch_k5_s1_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb1_left")
        self.comb1_right = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb1_right")

        self.comb2_left = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb2_left")

        self.comb3_left = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb3_left")
        self.comb3_right = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb3_right")

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb4_left")

    def call(self, x, x_prev, training=None):
        x_left = self.conv1x1(x, training=training)
        x_right = self.conv1x1_prev(x_prev, training=training)

        x0 = self.comb0_left(x_left, training=training) + self.comb0_right(x_right, training=training)
        x1 = self.comb1_left(x_right, training=training) + self.comb1_right(x_right, training=training)
        x2 = self.comb2_left(x_left, training=training) + x_right
        x3 = self.comb3_left(x_right, training=training) + self.comb3_right(x_right, training=training)
        x4 = self.comb4_left(x_left, training=training) + x_left

        x_out = tf.concat([x_right, x0, x1, x2, x3, x4], axis=get_channel_axis(self.data_format))
        return x_out


class ReductionBaseUnit(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 extra_padding=True,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionBaseUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.skip_input = True
        mid_channels = out_channels // 4

        self.conv1x1_prev = nas_conv1x1(
            in_channels=prev_in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1x1_prev")
        self.conv1x1 = nas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1x1")

        self.comb0_left = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb0_left")
        self.comb0_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb0_right")

        self.comb1_left = NasMaxPoolBlock(
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb1_left")
        self.comb1_right = dws_branch_k7_s2_p3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb1_right")

        self.comb2_left = NasAvgPoolBlock(
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb2_left")
        self.comb2_right = dws_branch_k5_s2_p2(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb2_right")

        self.comb3_right = nasnet_avgpool3x3_s1(
            data_format=data_format,
            name="comb3_right")

        self.comb4_left = dws_branch_k3_s1_p1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb4_left")
        self.comb4_right = NasMaxPoolBlock(
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb4_right")

    def call(self, x, x_prev, training=None):
        x_left = self.conv1x1(x, training=training)
        x_right = self.conv1x1_prev(x_prev, training=training)

        x0 = self.comb0_left(x_left, training=training) + self.comb0_right(x_right, training=training)
        x1 = self.comb1_left(x_left, training=training) + self.comb1_right(x_right, training=training)
        x2 = self.comb2_left(x_left, training=training) + self.comb2_right(x_right, training=training)
        x3 = x1 + self.comb3_right(x0, training=training)
        x4 = self.comb4_left(x0, training=training) + self.comb4_right(x_left, training=training)

        x_out = tf.concat([x1, x2, x3, x4], axis=get_channel_axis(self.data_format))
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(Reduction1Unit, self).__init__(
            in_channels=in_channels,
            prev_in_channels=prev_in_channels,
            out_channels=out_channels,
            extra_padding=True,
            data_format=data_format,
            **kwargs)


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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 extra_padding,
                 data_format="channels_last",
                 **kwargs):
        super(Reduction2Unit, self).__init__(
            in_channels=in_channels,
            prev_in_channels=prev_in_channels,
            out_channels=out_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            **kwargs)


class NASNetInitBlock(nn.Layer):
    """
    NASNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(NASNetInitBlock, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=2,
            padding=0,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = nasnet_batch_norm(
            channels=out_channels,
            data_format=data_format,
            name="bn")

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


class NASNet(tf.keras.Model):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(NASNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        reduction_units = [Reduction1Unit, Reduction2Unit]

        self.features = nasnet_dual_path_sequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=2,
            name="features")
        self.features.children.append(NASNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels

        out_channels = stem_blocks_channels[0]
        self.features.children.append(Stem1Unit(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="stem1_unit"))
        prev_in_channels = in_channels
        in_channels = out_channels

        out_channels = stem_blocks_channels[1]
        self.features.children.append(Stem2Unit(
            in_channels=in_channels,
            prev_in_channels=prev_in_channels,
            out_channels=out_channels,
            extra_padding=extra_padding,
            data_format=data_format,
            name="stem2_unit"))
        prev_in_channels = in_channels
        in_channels = out_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nasnet_dual_path_sequential(
                can_skip_input=skip_reduction_layer_input,
                name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                elif ((i == 0) and (j == 0)) or ((i != 0) and (j == 1)):
                    unit = FirstUnit
                else:
                    unit = NormalUnit
                if unit == Reduction2Unit:
                    stage.children.append(Reduction2Unit(
                        in_channels=in_channels,
                        prev_in_channels=prev_in_channels,
                        out_channels=out_channels,
                        extra_padding=extra_padding,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                else:
                    stage.children.append(unit(
                        in_channels=in_channels,
                        prev_in_channels=prev_in_channels,
                        out_channels=out_channels,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                prev_in_channels = in_channels
                in_channels = out_channels
            self.features.children.append(stage)

        self.features.children.append(nn.ReLU(name="activ"))
        self.features.children.append(nn.AveragePooling2D(
            pool_size=final_pool_size,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(nn.Dropout(
            rate=0.5,
            name="dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
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
               root=os.path.join("~", ".tensorflow", "models"),
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
    root : str, default '~/.tensorflow/models'
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
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def nasnet_4a1056(**kwargs):
    """
    NASNet-A 4@1056 (NASNet-A-Mobile) model from 'Learning Transferable Architectures for Scalable Image Recognition,'
    https://arxiv.org/abs/1707.07012.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
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
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        nasnet_4a1056,
        nasnet_6a4032,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 331, 331) if is_channels_first(data_format) else (batch, 331, 331, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nasnet_4a1056 or weight_count == 5289978)
        assert (model != nasnet_6a4032 or weight_count == 88753150)


if __name__ == "__main__":
    _test()
