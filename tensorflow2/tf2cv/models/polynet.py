"""
    PolyNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'PolyNet: A Pursuit of Structural Diversity in Very Deep Networks,'
    https://arxiv.org/abs/1611.05725.
"""

__all__ = ['PolyNet', 'polynet']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, Conv2d, ConvBlock, BatchNorm, SimpleSequential, ParametricSequential, Concurrent,\
    ParametricConcurrent, conv1x1_block, conv3x3_block, flatten, is_channels_first


class PolyConv(nn.Layer):
    """
    PolyNet specific convolution block. A block that is used inside poly-N (poly-2, poly-3, and so on) modules.
    The Convolution layer is shared between all Inception blocks inside a poly-N module. BatchNorm layers are not
    shared between Inception blocks and therefore the number of BatchNorm layers is equal to the number of Inception
    blocks inside a poly-N module.

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
    num_blocks : int
        Number of blocks (BatchNorm layers).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 num_blocks,
                 data_format="channels_last",
                 **kwargs):
        super(PolyConv, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bns = []
        for i in range(num_blocks):
            self.bns.append(BatchNorm(
                data_format=data_format,
                name="bn{}".format(i + 1)))
        self.activ = nn.ReLU()

    def call(self, x, index, training=None):
        x = self.conv(x)
        x = self.bns[index](x)
        x = self.activ(x)
        return x


def poly_conv1x1(in_channels,
                 out_channels,
                 num_blocks,
                 data_format="channels_last",
                 **kwargs):
    """
    1x1 version of the PolyNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_blocks : int
        Number of blocks (BatchNorm layers).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return PolyConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        num_blocks=num_blocks,
        data_format=data_format,
        **kwargs)


class MaxPoolBranch(nn.Layer):
    """
    PolyNet specific max pooling branch block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(MaxPoolBranch, self).__init__(**kwargs)
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.pool(x)
        return x


class Conv1x1Branch(nn.Layer):
    """
    PolyNet specific convolutional 1x1 branch block.

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
        super(Conv1x1Branch, self).__init__(**kwargs)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        return x


class Conv3x3Branch(nn.Layer):
    """
    PolyNet specific convolutional 3x3 branch block.

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
        super(Conv3x3Branch, self).__init__(**kwargs)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            padding=0,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        return x


class ConvSeqBranch(nn.Layer):
    """
    PolyNet specific convolutional sequence branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 data_format="channels_last",
                 **kwargs):
        super(ConvSeqBranch, self).__init__(**kwargs)
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = SimpleSequential(name="conv_list")
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add(ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                name="conv{}".format(i + 1)))
            in_channels = out_channels

    def call(self, x, training=None):
        x = self.conv_list(x, training=training)
        return x


class PolyConvSeqBranch(nn.Layer):
    """
    PolyNet specific convolutional sequence branch block with internal PolyNet specific convolution blocks.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    num_blocks : int
        Number of blocks for PolyConv.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 num_blocks,
                 data_format="channels_last",
                 **kwargs):
        super(PolyConvSeqBranch, self).__init__(**kwargs)
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = ParametricSequential(name="conv_list")
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add(PolyConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                num_blocks=num_blocks,
                data_format=data_format,
                name="conv{}".format(i + 1)))
            in_channels = out_channels

    def call(self, x, index, training=None):
        x = self.conv_list(x, index=index, training=training)
        return x


class TwoWayABlock(nn.Layer):
    """
    PolyNet type Inception-A block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(TwoWayABlock, self).__init__(**kwargs)
        in_channels = 384

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 48, 64),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            data_format=data_format,
            name="branch1"))
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 32),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            data_format=data_format,
            name="branch2"))
        self.branches.add(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=32,
            data_format=data_format,
            name="branch3"))
        self.conv = conv1x1_block(
            in_channels=128,
            out_channels=in_channels,
            activation=None,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        return x


class TwoWayBBlock(nn.Layer):
    """
    PolyNet type Inception-B block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(TwoWayBBlock, self).__init__(**kwargs)
        in_channels = 1152

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(128, 160, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            data_format=data_format,
            name="branch1"))
        self.branches.add(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=192,
            data_format=data_format,
            name="branch2"))
        self.conv = conv1x1_block(
            in_channels=384,
            out_channels=in_channels,
            activation=None,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        return x


class TwoWayCBlock(nn.Layer):
    """
    PolyNet type Inception-C block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(TwoWayCBlock, self).__init__(**kwargs)
        in_channels = 2048

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 3), (3, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 1), (1, 0)),
            data_format=data_format,
            name="branch1"))
        self.branches.add(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=192,
            data_format=data_format,
            name="branch2"))
        self.conv = conv1x1_block(
            in_channels=448,
            out_channels=in_channels,
            activation=None,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        return x


class PolyPreBBlock(nn.Layer):
    """
    PolyNet type PolyResidual-Pre-B block.

    Parameters:
    ----------
    num_blocks : int
        Number of blocks (BatchNorm layers).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 num_blocks,
                 data_format="channels_last",
                 **kwargs):
        super(PolyPreBBlock, self).__init__(**kwargs)
        in_channels = 1152

        self.branches = ParametricConcurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(PolyConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(128, 160, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            num_blocks=num_blocks,
            data_format=data_format,
            name="branch1"))
        self.branches.add(poly_conv1x1(
            in_channels=in_channels,
            out_channels=192,
            num_blocks=num_blocks,
            data_format=data_format,
            name="branch2"))

    def call(self, x, index, training=None):
        x = self.branches(x, index=index, training=training)
        return x


class PolyPreCBlock(nn.Layer):
    """
    PolyNet type PolyResidual-Pre-C block.

    Parameters:
    ----------
    num_blocks : int
        Number of blocks (BatchNorm layers).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 num_blocks,
                 data_format="channels_last",
                 **kwargs):
        super(PolyPreCBlock, self).__init__(**kwargs)
        in_channels = 2048

        self.branches = ParametricConcurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(PolyConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 3), (3, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 1), (1, 0)),
            num_blocks=num_blocks,
            data_format=data_format,
            name="branch1"))
        self.branches.add(poly_conv1x1(
            in_channels=in_channels,
            out_channels=192,
            num_blocks=num_blocks,
            data_format=data_format,
            name="branch2"))

    def call(self, x, index, training=None):
        x = self.branches(x, index=index, training=training)
        return x


def poly_res_b_block(data_format="channels_last",
                     **kwargs):
    """
    PolyNet type PolyResidual-Res-B block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return conv1x1_block(
        in_channels=384,
        out_channels=1152,
        strides=1,
        activation=None,
        data_format=data_format,
        **kwargs)


def poly_res_c_block(data_format="channels_last",
                     **kwargs):
    """
    PolyNet type PolyResidual-Res-C block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return conv1x1_block(
        in_channels=448,
        out_channels=2048,
        strides=1,
        activation=None,
        data_format=data_format,
        **kwargs)


class MultiResidual(nn.Layer):
    """
    Base class for constructing N-way modules (2-way, 3-way, and so on). Actually it is for 2-way modules.

    Parameters:
    ----------
    scale : float, default 1.0
        Scale value for each residual branch.
    res_block : HybridBlock class
        Residual branch block.
    num_blocks : int
        Number of residual branches.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 scale,
                 res_block,
                 num_blocks,
                 data_format="channels_last",
                 **kwargs):
        super(MultiResidual, self).__init__(**kwargs)
        assert (num_blocks >= 1)
        self.scale = scale
        self.num_blocks = num_blocks

        self.res_blocks = [res_block(
            data_format=data_format,
            name="res_block{}".format(i + 1)) for i in range(num_blocks)]
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        out = x
        for res_block in self.res_blocks:
            out = out + self.scale * res_block(x, training=training)
        out = self.activ(out)
        return out


class PolyResidual(nn.Layer):
    """
    The other base class for constructing N-way poly-modules. Actually it is for 3-way poly-modules.

    Parameters:
    ----------
    scale : float, default 1.0
        Scale value for each residual branch.
    res_block : HybridBlock class
        Residual branch block.
    num_blocks : int
        Number of residual branches.
    pre_block : HybridBlock class
        Preliminary block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 scale,
                 res_block,
                 num_blocks,
                 pre_block,
                 data_format="channels_last",
                 **kwargs):
        super(PolyResidual, self).__init__(**kwargs)
        assert (num_blocks >= 1)
        self.scale = scale

        self.pre_block = pre_block(
            num_blocks=num_blocks,
            data_format=data_format,
            name="pre_block")
        self.res_blocks = [res_block(
            data_format=data_format,
            name="res_block{}".format(i + 1)) for i in range(num_blocks)]
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        out = x
        for index, res_block in enumerate(self.res_blocks):
            x = self.pre_block(x, index, training=training)
            x = res_block(x, training=training)
            out = out + self.scale * x
            x = self.activ(x)
        out = self.activ(out)
        return out


class PolyBaseUnit(nn.Layer):
    """
    PolyNet unit base class.

    Parameters:
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    two_way_block : HybridBlock class
        Residual branch block for 2-way-stage.
    poly_scale : float, default 0.0
        Scale value for 2-way stage.
    poly_res_block : HybridBlock class, default None
        Residual branch block for poly-stage.
    poly_pre_block : HybridBlock class, default None
        Preliminary branch block for poly-stage.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 two_way_scale,
                 two_way_block,
                 poly_scale=0.0,
                 poly_res_block=None,
                 poly_pre_block=None,
                 data_format="channels_last",
                 **kwargs):
        super(PolyBaseUnit, self).__init__(**kwargs)

        if poly_res_block is not None:
            assert (poly_scale != 0.0)
            assert (poly_pre_block is not None)
            self.poly = PolyResidual(
                scale=poly_scale,
                res_block=poly_res_block,
                num_blocks=3,
                pre_block=poly_pre_block,
                data_format=data_format,
                name="poly")
        else:
            assert (poly_scale == 0.0)
            assert (poly_pre_block is None)
            self.poly = None
        self.twoway = MultiResidual(
            scale=two_way_scale,
            res_block=two_way_block,
            num_blocks=2,
            data_format=data_format,
            name="twoway")

    def call(self, x, training=None):
        if self.poly is not None:
            x = self.poly(x, training=training)
        x = self.twoway(x, training=training)
        return x


class PolyAUnit(PolyBaseUnit):
    """
    PolyNet type A unit.

    Parameters:
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    poly_scale : float
        Scale value for 2-way stage.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 two_way_scale,
                 poly_scale=0.0,
                 data_format="channels_last",
                 **kwargs):
        super(PolyAUnit, self).__init__(
            two_way_scale=two_way_scale,
            two_way_block=TwoWayABlock,
            data_format=data_format,
            **kwargs)
        assert (poly_scale == 0.0)


class PolyBUnit(PolyBaseUnit):
    """
    PolyNet type B unit.

    Parameters:
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    poly_scale : float
        Scale value for 2-way stage.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 two_way_scale,
                 poly_scale,
                 data_format="channels_last",
                 **kwargs):
        super(PolyBUnit, self).__init__(
            two_way_scale=two_way_scale,
            two_way_block=TwoWayBBlock,
            poly_scale=poly_scale,
            poly_res_block=poly_res_b_block,
            poly_pre_block=PolyPreBBlock,
            data_format=data_format,
            **kwargs)


class PolyCUnit(PolyBaseUnit):
    """
    PolyNet type C unit.

    Parameters:
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    poly_scale : float
        Scale value for 2-way stage.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 two_way_scale,
                 poly_scale,
                 data_format="channels_last",
                 **kwargs):
        super(PolyCUnit, self).__init__(
            two_way_scale=two_way_scale,
            two_way_block=TwoWayCBlock,
            poly_scale=poly_scale,
            poly_res_block=poly_res_c_block,
            poly_pre_block=PolyPreCBlock,
            data_format=data_format,
            **kwargs)


class ReductionAUnit(nn.Layer):
    """
    PolyNet type Reduction-A unit.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionAUnit, self).__init__(**kwargs)
        in_channels = 384

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 384),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            data_format=data_format,
            name="branch1"))
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,),
            data_format=data_format,
            name="branch2"))
        self.branches.add(MaxPoolBranch(
            data_format=data_format,
            name="branch3"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class ReductionBUnit(nn.Layer):
    """
    PolyNet type Reduction-B unit.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionBUnit, self).__init__(**kwargs)
        in_channels = 1152

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 256),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            data_format=data_format,
            name="branch1"))
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            data_format=data_format,
            name="branch2"))
        self.branches.add(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 384),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            data_format=data_format,
            name="branch3"))
        self.branches.add(MaxPoolBranch(
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class PolyBlock3a(nn.Layer):
    """
    PolyNet type Mixed-3a block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(PolyBlock3a, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(MaxPoolBranch(
            data_format=data_format,
            name="branch1"))
        self.branches.add(Conv3x3Branch(
            in_channels=64,
            out_channels=96,
            data_format=data_format,
            name="branch2"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class PolyBlock4a(nn.Layer):
    """
    PolyNet type Mixed-4a block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(PolyBlock4a, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 0),
            data_format=data_format,
            name="branch1"))
        self.branches.add(ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 64, 64, 96),
            kernel_size_list=(1, (7, 1), (1, 7), 3),
            strides_list=(1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), 0),
            data_format=data_format,
            name="branch2"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class PolyBlock5a(nn.Layer):
    """
    PolyNet type Mixed-5a block.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(PolyBlock5a, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(MaxPoolBranch(
            data_format=data_format,
            name="branch1"))
        self.branches.add(Conv3x3Branch(
            in_channels=192,
            out_channels=192,
            data_format=data_format,
            name="branch2"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class PolyInitBlock(nn.Layer):
    """
    PolyNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 data_format="channels_last",
                 **kwargs):
        super(PolyInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            strides=2,
            padding=0,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            padding=0,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=32,
            out_channels=64,
            data_format=data_format,
            name="conv3")
        self.block1 = PolyBlock3a(
            data_format=data_format,
            name="block1")
        self.block2 = PolyBlock4a(
            data_format=data_format,
            name="block2")
        self.block3 = PolyBlock5a(
            data_format=data_format,
            name="block3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        return x


class PolyNet(tf.keras.Model):
    """
    PolyNet model from 'PolyNet: A Pursuit of Structural Diversity in Very Deep Networks,'
    https://arxiv.org/abs/1611.05725.

    Parameters:
    ----------
    two_way_scales : list of list of floats
        Two way scale values for each normal unit.
    poly_scales : list of list of floats
        Three way scale values for each normal unit.
    dropout_rate : float, default 0.2
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (331, 331)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 two_way_scales,
                 poly_scales,
                 dropout_rate=0.2,
                 in_channels=3,
                 in_size=(331, 331),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(PolyNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        normal_units = [PolyAUnit, PolyBUnit, PolyCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = SimpleSequential(name="features")
        self.features.add(PolyInitBlock(
            in_channels=in_channels,
            data_format=data_format,
            name="init_block"))

        for i, (two_way_scales_per_stage, poly_scales_per_stage) in enumerate(zip(two_way_scales, poly_scales)):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, (two_way_scale, poly_scale) in enumerate(zip(two_way_scales_per_stage, poly_scales_per_stage)):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                    stage.add(unit(
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                else:
                    unit = normal_units[i]
                    stage.add(unit(
                        two_way_scale=two_way_scale,
                        poly_scale=poly_scale,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=9,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(nn.Dropout(
            rate=dropout_rate,
            name="dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=2048,
            name="fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_polynet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create PolyNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    two_way_scales = [
        [1.000000, 0.992308, 0.984615, 0.976923, 0.969231, 0.961538, 0.953846, 0.946154, 0.938462, 0.930769],
        [0.000000, 0.915385, 0.900000, 0.884615, 0.869231, 0.853846, 0.838462, 0.823077, 0.807692, 0.792308, 0.776923],
        [0.000000, 0.761538, 0.746154, 0.730769, 0.715385, 0.700000]]
    poly_scales = [
        [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        [0.000000, 0.923077, 0.907692, 0.892308, 0.876923, 0.861538, 0.846154, 0.830769, 0.815385, 0.800000, 0.784615],
        [0.000000, 0.769231, 0.753846, 0.738462, 0.723077, 0.707692]]

    net = PolyNet(
        two_way_scales=two_way_scales,
        poly_scales=poly_scales,
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


def polynet(**kwargs):
    """
    PolyNet model from 'PolyNet: A Pursuit of Structural Diversity in Very Deep Networks,'
    https://arxiv.org/abs/1611.05725.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_polynet(model_name="polynet", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        polynet,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 331, 331) if is_channels_first(data_format) else (batch, 331, 331, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != polynet or weight_count == 95366600)


if __name__ == "__main__":
    _test()
