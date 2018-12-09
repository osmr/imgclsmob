"""
    Common routines for models in Gluon.
"""

__all__ = ['ReLU6', 'conv1x1', 'ConvBlock', 'conv1x1_block', 'conv3x3_block', 'conv7x7_block', 'dwconv3x3_block',
           'PreConvBlock', 'pre_conv1x1_block', 'pre_conv3x3_block', 'ChannelShuffle', 'ChannelShuffle2', 'SEBlock',
           'IBN', 'DualPathSequential', 'ParametricSequential', 'ParametricConcurrent', 'Hourglass']

import math
from mxnet.gluon import nn, HybridBlock


class ReLU6(nn.HybridBlock):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


def conv1x1(in_channels,
            out_channels,
            strides=1,
            use_bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        strides=strides,
        use_bias=use_bias,
        in_channels=in_channels)


class ConvBlock(HybridBlock):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 bn_use_global_stats=False,
                 act_type="relu",
                 activate=True,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.activate = activate

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            if self.activate:
                if act_type == "relu6":
                    self.activ = ReLU6()
                else:
                    self.activ = nn.Activation(act_type)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  strides=1,
                  groups=1,
                  use_bias=False,
                  bn_use_global_stats=False,
                  act_type="relu",
                  activate=True):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        act_type=act_type,
        activate=activate)


def conv3x3_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  bn_use_global_stats=False,
                  act_type="relu",
                  activate=True):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        act_type=act_type,
        activate=activate)


def conv7x7_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  bn_use_global_stats=False,
                  act_type="relu",
                  activate=True):
    """
    7x7 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        act_type=act_type,
        activate=activate)


def dwconv3x3_block(in_channels,
                    out_channels,
                    strides,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    bn_use_global_stats=False,
                    act_type="relu",
                    activate=True):
    """
    3x3 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return conv3x3_block(
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        act_type=act_type,
        activate=activate)


class PreConvBlock(HybridBlock):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

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
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 bn_use_global_stats,
                 return_preact=False,
                 **kwargs):
        super(PreConvBlock, self).__init__(**kwargs)
        self.return_preact = return_preact

        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv1x1_block(in_channels,
                      out_channels,
                      strides=1,
                      bn_use_global_stats=False,
                      return_preact=False):
    """
    1x1 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      strides=1,
                      bn_use_global_stats=False,
                      return_preact=False):
    """
    3x3 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact)


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : NDArray
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    NDArray
        Resulted tensor.
    """
    return x.reshape((0, -4, groups, -1, -2)).swapaxes(1, 2).reshape((0, -3, -2))


class ChannelShuffle(HybridBlock):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups,
                 **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.groups = groups

    def hybrid_forward(self, F, x):
        return channel_shuffle(x, self.groups)


def channel_shuffle2(x,
                     channels_per_group):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.

    Parameters:
    ----------
    x : NDArray
        Input tensor.
    channels_per_group : int
        Number of channels per group.

    Returns
    -------
    NDArray
        Resulted tensor.
    """
    return x.reshape((0, -4, channels_per_group, -1, -2)).swapaxes(1, 2).reshape((0, -3, -2))


class ChannelShuffle2(HybridBlock):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups,
                 **kwargs):
        super(ChannelShuffle2, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.channels_per_group = channels // groups

    def hybrid_forward(self, F, x):
        return channel_shuffle2(x, self.channels_per_group)


class SEBlock(HybridBlock):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        mid_cannels = channels // reduction

        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_cannels,
                use_bias=True)
            self.relu = nn.Activation('relu')
            self.conv2 = conv1x1(
                in_channels=mid_cannels,
                out_channels=channels,
                use_bias=True)
            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = F.broadcast_mul(x, w)
        return x


def split(x,
          sizes,
          axis=1):
    """
    Splits an array along a particular axis into multiple sub-arrays.

    Parameters:
    ----------
    x : NDArray/symbol
        Input tensor.
    sizes : tuple/list of int
        Sizes of chunks.
    axis : int, default 1
        Axis along which to split.

    Returns
    -------
    Tuple of NDArray/symbol
        Resulted tensor.
    """
    x_outs = []
    begin = 0
    for size in sizes:
        end = begin + size
        x_outs += [x.slice_axis(axis=axis, begin=begin, end=end)]
        begin = end
    return tuple(x_outs)


class IBN(HybridBlock):
    """
    Instance-Batch Normalization block from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    channels : int
        Number of channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    inst_fraction : float, default 0.5
        The first fraction of channels for normalization.
    inst_first : bool, default True
        Whether instance normalization be on the first part of channels.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats=False,
                 first_fraction=0.5,
                 inst_first=True,
                 **kwargs):
        super(IBN, self).__init__(**kwargs)
        self.inst_first = inst_first
        h1_channels = int(math.floor(channels * first_fraction))
        h2_channels = channels - h1_channels
        self.split_sections = [h1_channels, h2_channels]

        if self.inst_first:
            self.inst_norm = nn.InstanceNorm(
                in_channels=h1_channels,
                scale=True)
            self.batch_norm = nn.BatchNorm(
                in_channels=h2_channels,
                use_global_stats=bn_use_global_stats)

        else:
            self.batch_norm = nn.BatchNorm(
                in_channels=h1_channels,
                use_global_stats=bn_use_global_stats)
            self.inst_norm = nn.InstanceNorm(
                in_channels=h2_channels,
                scale=True)

    def hybrid_forward(self, F, x):
        x1, x2 = split(x, sizes=self.split_sections, axis=1)
        if self.inst_first:
            x1 = self.inst_norm(x1)
            x2 = self.batch_norm(x2)
        else:
            x1 = self.batch_norm(x1)
            x2 = self.inst_norm(x2)
        x = F.concat(x1, x2, dim=1)
        return x


class DualPathSequential(nn.HybridSequential):
    """
    A sequential container for hybrid blocks with dual inputs/outputs.
    Blocks will be executed in the order they are added.

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
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda block, x1, x2: block(x1, x2)),
                 dual_path_scheme_ordinal=(lambda block, x1, x2: (block(x1), x2)),
                 **kwargs):
        super(DualPathSequential, self).__init__(**kwargs)
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def hybrid_forward(self, F, x1, x2=None):
        length = len(self._children.values())
        for i, block in enumerate(self._children.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(block, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(block, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class ParametricSequential(nn.HybridSequential):
    """
    A sequential container for modules with parameters.
    Blocks will be executed in the order they are added.
    """
    def __init__(self,
                 **kwargs):
        super(ParametricSequential, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        for block in self._children.values():
            x = block(x, *args, **kwargs)
        return x


class ParametricConcurrent(nn.HybridSequential):
    """
    A container for concatenation of modules with parameters.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self,
                 axis=1,
                 **kwargs):
        super(ParametricConcurrent, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = []
        for block in self._children.values():
            out.append(block(x, *args, **kwargs))
        out = F.concat(*out, dim=self.axis)
        return out


class Hourglass(HybridBlock):
    """
    A hourglass block.

    Parameters:
    ----------
    down_seq : nn.HybridSequential
        Down modules as sequential.
    up_seq : nn.HybridSequential
        Up modules as sequential.
    skip_seq : nn.HybridSequential
        Skip connection modules as sequential.
    merge_type : str, default 'add'
        Type of concatenation of up and skip outputs.
    return_first_skip : bool, default False
        Whether return the first skip connection output. Used in ResAttNet.
    """
    def __init__(self,
                 down_seq,
                 up_seq,
                 skip_seq,
                 merge_type="add",
                 return_first_skip=False,
                 **kwargs):
        super(Hourglass, self).__init__(**kwargs)
        assert (len(up_seq) == len(down_seq))
        assert (len(skip_seq) == len(down_seq))
        assert (merge_type in ["add"])
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.depth = len(down_seq)

        with self.name_scope():
            self.down_seq = down_seq
            self.up_seq = up_seq
            self.skip_seq = skip_seq

    def hybrid_forward(self, F, x):
        y = None
        down_outs = [x]
        for down_module in self.down_seq._children.values():
            x = down_module(x)
            down_outs.append(x)
        for i in range(len(down_outs)):
            if i != 0:
                y = down_outs[self.depth - i]
                skip_module = self.skip_seq[self.depth - i]
                y = skip_module(y)
                if (y is not None) and (self.merge_type == "add"):
                    x = x + y
            if i != len(down_outs) - 1:
                up_module = self.up_seq[self.depth - 1 - i]
                x = up_module(x)
        if self.return_first_skip:
            return x, y
        else:
            return x
