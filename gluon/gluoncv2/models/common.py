"""
    Common routines for models in Gluon.
"""

__all__ = ['ReLU6', 'PReLU2', 'HSwish', 'conv1x1', 'conv3x3', 'depthwise_conv3x3', 'ConvBlock', 'conv1x1_block',
           'conv3x3_block', 'conv7x7_block', 'dwconv3x3_block', 'dwconv5x5_block', 'PreConvBlock', 'pre_conv1x1_block',
           'pre_conv3x3_block', 'ChannelShuffle', 'ChannelShuffle2', 'SEBlock', 'IBN', 'DualPathSequential',
           'ParametricSequential', 'Concurrent', 'ParametricConcurrent', 'Hourglass', 'SesquialteralHourglass',
           'MultiOutputSequential']

import math
from inspect import isfunction
import mxnet as mx
from mxnet.gluon import nn, HybridBlock


class ReLU6(HybridBlock):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0.0, 6.0, name="relu6")


class PReLU2(HybridBlock):
    """
    Parametric leaky version of a Rectified Linear Unit (with wide alpha).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    alpha_initializer : Initializer
        Initializer for the `embeddings` matrix.
    """
    def __init__(self,
                 in_channels=1,
                 alpha_initializer=mx.init.Constant(0.25),
                 **kwargs):
        super(PReLU2, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get("alpha", shape=(in_channels,), init=alpha_initializer)

    def hybrid_forward(self, F, x, alpha):
        return F.LeakyReLU(x, gamma=alpha, act_type="prelu", name="fwd")


class HSigmoid(HybridBlock):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0


class HSwish(HybridBlock):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x * F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or HybridBlock
        Activation function or name of activation function.

    Returns
    -------
    HybridBlock
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu6":
            return ReLU6()
        elif activation == "swish":
            return nn.Swish()
        elif activation == "hswish":
            return HSwish()
        else:
            return nn.Activation(activation)
    else:
        assert (isinstance(activation, HybridBlock))
        return activation


def conv1x1(in_channels,
            out_channels,
            strides=1,
            groups=1,
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
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        in_channels=in_channels)


def conv3x3(in_channels,
            out_channels,
            strides=1,
            padding=1,
            dilation=1,
            groups=1,
            use_bias=False):
    """
    Convolution 3x3 layer.

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
    """
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        in_channels=in_channels)


def depthwise_conv3x3(channels,
                      strides):
    """
    Depthwise convolution 3x3 layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return nn.Conv2D(
        channels=channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=channels,
        use_bias=False,
        in_channels=channels)


class ConvBlock(HybridBlock):
    """
    Standard convolution block with Batch normalization and activation.

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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
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
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

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
            if self.use_bn:
                self.bn = nn.BatchNorm(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  strides=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


def conv3x3_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
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
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


def conv5x5_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=2,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
    """
    5x5 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 2
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


def conv7x7_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  use_bn=True,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


def dwconv3x3_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    bn_epsilon=1e-5,
                    bn_use_global_stats=False,
                    activation=(lambda: nn.Activation("relu")),
                    **kwargs):
    """
    3x3 depthwise version of the standard convolution block.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return conv3x3_block(
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


def dwconv5x5_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=2,
                    dilation=1,
                    use_bias=False,
                    bn_epsilon=1e-5,
                    bn_use_global_stats=False,
                    activation=(lambda: nn.Activation("relu")),
                    **kwargs):
    """
    5x5 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 2
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return conv5x5_block(
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
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
                 return_preact=False,
                 activate=True,
                 **kwargs):
        super(PreConvBlock, self).__init__(**kwargs)
        self.return_preact = return_preact
        self.activate = activate

        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        if self.activate:
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
                      use_bias=False,
                      bn_use_global_stats=False,
                      return_preact=False,
                      activate=True):
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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        activate=activate)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      bn_use_global_stats=False,
                      return_preact=False,
                      activate=True):
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
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        activate=activate)


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : Symbol or NDArray
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    Symbol or NDArray
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
    x : Symbol or NDArray
        Input tensor.
    channels_per_group : int
        Number of channels per group.

    Returns
    -------
    Symbol or NDArray
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
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    activation : function, or str, or HybridBlock
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 approx_sigmoid=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        mid_cannels = channels // reduction

        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_cannels,
                use_bias=True)
            self.activ = get_activation_layer(activation)
            self.conv2 = conv1x1(
                in_channels=mid_cannels,
                out_channels=channels,
                use_bias=True)
            self.sigmoid = HSigmoid() if approx_sigmoid else nn.Activation("sigmoid")

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.conv1(w)
        w = self.activ(w)
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
    x : Symbol or NDArray
        Input tensor.
    sizes : tuple/list of int
        Sizes of chunks.
    axis : int, default 1
        Axis along which to split.

    Returns
    -------
    Tuple of Symbol or NDArray
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
    A sequential container for blocks with parameters.
    Blocks will be executed in the order they are added.
    """
    def __init__(self,
                 **kwargs):
        super(ParametricSequential, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        for block in self._children.values():
            x = block(x, *args, **kwargs)
        return x


class Concurrent(nn.HybridSequential):
    """
    A container for concatenation of blocks on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    """
    def __init__(self,
                 axis=1,
                 stack=False,
                 **kwargs):
        super(Concurrent, self).__init__(**kwargs)
        self.axis = axis
        self.stack = stack

    def hybrid_forward(self, F, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        if self.stack:
            out = F.stack(*out, axis=self.axis)
        else:
            out = F.concat(*out, dim=self.axis)
        return out


class ParametricConcurrent(nn.HybridSequential):
    """
    A container for concatenation of blocks with parameters.

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


class SesquialteralHourglass(HybridBlock):
    """
    A sesquialteral hourglass block.

    Parameters:
    ----------
    down1_seq : nn.Sequential
        The first down modules as sequential.
    skip1_seq : nn.Sequential
        The first skip connection modules as sequential.
    up_seq : nn.Sequential
        Up modules as sequential.
    skip2_seq : nn.Sequential
        The second skip connection modules as sequential.
    down2_seq : nn.Sequential
        The second down modules as sequential.
    merge_type : str, default 'con'
        Type of concatenation of up and skip outputs.
    """
    def __init__(self,
                 down1_seq,
                 skip1_seq,
                 up_seq,
                 skip2_seq,
                 down2_seq,
                 merge_type="cat",
                 **kwargs):
        super(SesquialteralHourglass, self).__init__(**kwargs)
        assert (len(down1_seq) == len(up_seq))
        assert (len(down1_seq) == len(down2_seq))
        assert (len(skip1_seq) == len(skip2_seq))
        assert (len(down1_seq) == len(skip1_seq) - 1)
        assert (merge_type in ["cat", "add"])
        self.merge_type = merge_type
        self.depth = len(down1_seq)

        with self.name_scope():
            self.down1_seq = down1_seq
            self.skip1_seq = skip1_seq
            self.up_seq = up_seq
            self.skip2_seq = skip2_seq
            self.down2_seq = down2_seq

    def _merge(self, F, x, y):
        if y is not None:
            if self.merge_type == "cat":
                x = F.concat(x, y, dim=1)
            elif self.merge_type == "add":
                x = x + y
        return x

    def hybrid_forward(self, F, x):
        y = self.skip1_seq[0](x)
        skip1_outs = [y]
        for i in range(self.depth):
            x = self.down1_seq[i](x)
            y = self.skip1_seq[i + 1](x)
            skip1_outs.append(y)
        x = skip1_outs[self.depth]
        y = self.skip2_seq[0](x)
        skip2_outs = [y]
        for i in range(self.depth):
            x = self.up_seq[i](x)
            y = skip1_outs[self.depth - 1 - i]
            x = self._merge(F, x, y)
            y = self.skip2_seq[i + 1](x)
            skip2_outs.append(y)
        x = self.skip2_seq[self.depth](x)
        for i in range(self.depth):
            x = self.down2_seq[i](x)
            y = skip2_outs[self.depth - 1 - i]
            x = self._merge(F, x, y)
        return x


class MultiOutputSequential(nn.HybridSequential):
    """
    A sequential container with multiple outputs.
    Blocks will be executed in the order they are added.
    """
    def __init__(self,
                 **kwargs):
        super(MultiOutputSequential, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        outs = []
        for block in self._children.values():
            x = block(x)
            if hasattr(block, "do_output") and block.do_output:
                outs.append(x)
        return [x] + outs
