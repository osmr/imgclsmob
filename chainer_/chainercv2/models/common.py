"""
    Common routines for models in Chainer.
"""

__all__ = ['round_channels', 'get_activation_layer', 'ReLU6', 'HSwish', 'GlobalAvgPool2D', 'SelectableDense',
           'DenseBlock', 'ConvBlock1d', 'conv1x1', 'conv3x3', 'depthwise_conv3x3', 'ConvBlock', 'conv1x1_block',
           'conv3x3_block', 'conv7x7_block', 'dwconv_block', 'dwconv3x3_block', 'dwconv5x5_block', 'dwsconv3x3_block',
           'PreConvBlock', 'pre_conv1x1_block', 'pre_conv3x3_block', 'DeconvBlock', 'ChannelShuffle', 'ChannelShuffle2',
           'SEBlock', 'SABlock', 'SAConvBlock', 'saconv3x3_block', 'PixelShuffle', 'DucBlock', 'SimpleSequential',
           'DualPathSequential', 'Concurrent', 'SequentialConcurrent', 'ParametricSequential', 'ParametricConcurrent',
           'Hourglass', 'SesquialteralHourglass', 'MultiOutputSequential', 'ParallelConcurent', 'Flatten',
           'AdaptiveAvgPool2D', 'InterpolationBlock', 'HeatmapMaxDetBlock']

from inspect import isfunction
from functools import partial
import numpy as np
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer import link
from chainer.initializers import _get_initializer
from chainer.variable import Parameter


def round_channels(channels,
                   divisor=8):
    """
    Round weighted channel number (make divisible operation).

    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns:
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


class ReLU6(Chain):
    """
    ReLU6 activation layer.
    """
    def __call__(self, x):
        return F.clip(x, 0.0, 6.0)


class Swish(Chain):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def __call__(self, x):
        return x * F.sigmoid(x)


class HSigmoid(Chain):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __call__(self, x):
        return F.clip(x + 3.0, 0.0, 6.0) / 6.0


class HSwish(Chain):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __call__(self, x):
        return x * F.clip(x + 3.0, 0.0, 6.0) / 6.0


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function or str
        Activation function or name of activation function.

    Returns:
    -------
    function
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return F.relu
        elif activation == "relu6":
            return ReLU6()
        elif activation == "swish":
            return Swish()
            # return partial(
            #     F.swish,
            #     beta=[1.0])
        elif activation == "hswish":
            return HSwish()
        elif activation == "sigmoid":
            return F.sigmoid
        elif activation == "hsigmoid":
            return HSigmoid()
        else:
            raise NotImplementedError()
    else:
        return activation


class GlobalAvgPool2D(Chain):
    """
    Global average pooling operation for spatial data.
    """
    def __call__(self, x):
        return F.average_pooling_2d(x, ksize=x.shape[2:])


class SelectableDense(link.Link):
    """
    Selectable dense layer.

    Parameters:
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    initial_weight : `types.InitializerSpec`, default None
        Initializer for the `kernel` weights matrix.
    initial_bias: `types.InitializerSpec`, default 0
        Initializer for the bias vector.
    num_options : int, default 1
        Number of selectable options.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias=False,
                 initial_weight=None,
                 initial_bias=0,
                 num_options=1):
        super(SelectableDense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.num_options = num_options

        with self.init_scope():
            weight_initializer = _get_initializer(initial_weight)
            self.weight = Parameter(
                initializer=weight_initializer,
                shape=(num_options, out_channels, in_channels),
                name="weight")
            if use_bias:
                bias_initializer = _get_initializer(initial_bias)
                self.bias = Parameter(
                    initializer=bias_initializer,
                    shape=(num_options, out_channels),
                    name="bias")
            else:
                self.bias = None

    def forward(self, x, indices):
        weight = self.xp.take(self.weight.data, indices=indices, axis=0)
        x = F.expand_dims(x, axis=-1)
        x = F.batch_matmul(weight, x)
        x = F.squeeze(x, axis=-1)
        if self.use_bias:
            bias = self.xp.take(self.bias.data, indices=indices, axis=0)
            x += bias
        return x

    @property
    def printable_specs(self):
        specs = [
            ('in_channels', self.in_channels),
            ('out_channels', self.out_channels),
            ('use_bias', self.use_bias),
            ('num_options', self.num_options),
        ]
        for spec in specs:
            yield spec


class DenseBlock(Chain):
    """
    Standard dense block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu)):
        super(DenseBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.init_scope():
            self.fc = L.Linear(
                in_size=in_channels,
                out_size=out_channels,
                nobias=(not use_bias))
            if self.use_bn:
                self.bn = L.BatchNormalization(
                    size=out_channels,
                    eps=bn_eps)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def __call__(self, x):
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class ConvBlock1d(Chain):
    """
    Standard 1D convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int
        Convolution window size.
    stride : int
        Stride of the convolution.
    pad : int
        Padding value for convolution layer.
    dilate : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu)):
        super(ConvBlock1d, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.init_scope():
            self.conv = L.Convolution1D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
                groups=groups)
            if self.use_bn:
                self.bn = L.BatchNormalization(
                    size=out_channels,
                    eps=bn_eps)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def __call__(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1(in_channels,
            out_channels,
            stride=1,
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
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return L.Convolution2D(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        nobias=(not use_bias),
        groups=groups)


def conv3x3(in_channels,
            out_channels,
            stride=1,
            pad=1,
            dilate=1,
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
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return L.Convolution2D(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        nobias=(not use_bias),
        dilate=dilate,
        groups=groups)


def depthwise_conv3x3(channels,
                      stride):
    """
    Depthwise convolution 3x3 layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    return L.Convolution2D(
        in_channels=channels,
        out_channels=channels,
        ksize=3,
        stride=stride,
        pad=1,
        nobias=True,
        groups=channels)


class ConvBlock(Chain):
    """
    Standard convolution block with Batch normalization and activation.

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
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu)):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
                groups=groups)
            if self.use_bn:
                self.bn = L.BatchNormalization(
                    size=out_channels,
                    eps=bn_eps)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def __call__(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: F.relu)):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=0,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  pad=1,
                  dilate=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: F.relu)):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def conv5x5_block(in_channels,
                  out_channels,
                  stride=1,
                  pad=2,
                  dilate=1,
                  groups=1,
                  use_bias=False,
                  bn_eps=1e-5,
                  activation=(lambda: F.relu)):
    """
    5x5 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 2
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=5,
        stride=stride,
        pad=pad,
        dilate=dilate,
        groups=groups,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation)


def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  pad=3,
                  use_bias=False,
                  use_bn=True,
                  activation=(lambda: F.relu)):
    """
    7x7 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=7,
        stride=stride,
        pad=pad,
        use_bias=use_bias,
        use_bn=use_bn,
        activation=activation)


def dwconv_block(in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu)):
    """
    Depthwise version of the standard convolution block.

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
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=ksize,
        stride=stride,
        pad=pad,
        dilate=dilate,
        groups=out_channels,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def dwconv3x3_block(in_channels,
                    out_channels,
                    stride=1,
                    pad=1,
                    dilate=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation=(lambda: F.relu)):
    """
    3x3 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation)


def dwconv5x5_block(in_channels,
                    out_channels,
                    stride=1,
                    pad=2,
                    dilate=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation=(lambda: F.relu)):
    """
    5x5 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 2
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=5,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation)


class DwsConvBlock(Chain):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default F.relu
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default F.relu
        Activation function after the pointwise convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_eps=1e-5,
                 dw_activation=(lambda: F.relu),
                 pw_activation=(lambda: F.relu)):
        super(DwsConvBlock, self).__init__()
        with self.init_scope():
            self.dw_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                use_bias=use_bias,
                use_bn=dw_use_bn,
                bn_eps=bn_eps,
                activation=dw_activation)
            self.pw_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=pw_use_bn,
                bn_eps=bn_eps,
                activation=pw_activation)

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     stride=1,
                     pad=1,
                     dilate=1,
                     use_bias=False,
                     bn_eps=1e-5,
                     dw_activation=(lambda: F.relu),
                     pw_activation=(lambda: F.relu),
                     **kwargs):
    """
    3x3 depthwise separable version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default F.relu
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default F.relu
        Activation function after the pointwise convolution block.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        **kwargs)


class PreConvBlock(Chain):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

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
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 use_bn=True,
                 return_preact=False,
                 activate=True):
        super(PreConvBlock, self).__init__()
        self.return_preact = return_preact
        self.activate = activate
        self.use_bn = use_bn

        with self.init_scope():
            if self.use_bn:
                self.bn = L.BatchNormalization(
                    size=in_channels,
                    eps=1e-5)
            if self.activate:
                self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate)

    def __call__(self, x):
        if self.use_bn:
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
                      stride=1,
                      use_bias=False,
                      use_bn=True,
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
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=0,
        use_bias=use_bias,
        use_bn=use_bn,
        return_preact=return_preact,
        activate=activate)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      stride=1,
                      pad=1,
                      dilate=1,
                      use_bias=False,
                      use_bn=True,
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
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        use_bn=use_bn,
        return_preact=return_preact,
        activate=activate)


class DeconvBlock(Chain):
    """
    Deconvolution block with batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the deconvolution.
    pad : int or tuple/list of 2 int
        Padding value for deconvolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu),
                 **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.init_scope():
            self.conv = L.Deconvolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
                groups=groups)
            if self.use_bn:
                self.bn = L.BatchNormalization(
                    size=out_channels,
                    eps=bn_eps)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def __call__(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : chainer.Variable or numpy.ndarray or cupy.ndarray
        Input variable.
    groups : int
        Number of groups.

    Returns:
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted variable.
    """
    batch, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = F.reshape(x, shape=(batch, groups, channels_per_group, height, width))
    x = F.swapaxes(x, axis1=1, axis2=2)
    x = F.reshape(x, shape=(batch, channels, height, width))
    return x


class ChannelShuffle(Chain):
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
                 groups):
        super(ChannelShuffle, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def __call__(self, x):
        return channel_shuffle(x, self.groups)


def channel_shuffle2(x,
                     groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.

    Parameters:
    ----------
    x : chainer.Variable or numpy.ndarray or cupy.ndarray
        Input variable.
    groups : int
        Number of groups.

    Returns:
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted variable.
    """
    batch, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = F.reshape(x, shape=(batch, channels_per_group, groups, height, width))
    x = F.swapaxes(x, axis1=1, axis2=2)
    x = F.reshape(x, shape=(batch, channels, height, width))
    return x


class ChannelShuffle2(Chain):
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
                 groups):
        super(ChannelShuffle2, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def __call__(self, x):
        return channel_shuffle2(x, self.groups)


class SEBlock(Chain):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function or str, default F.relu
        Activation function after the first convolution.
    out_activation : function or str, default F.sigmoid
        Activation function after the last convolution.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 mid_channels=None,
                 round_mid=False,
                 use_conv=True,
                 mid_activation=(lambda: F.relu),
                 out_activation=(lambda: F.sigmoid)):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        with self.init_scope():
            if use_conv:
                self.conv1 = conv1x1(
                    in_channels=channels,
                    out_channels=mid_channels,
                    use_bias=True)
            else:
                self.fc1 = L.Linear(
                    in_size=channels,
                    out_size=mid_channels)
            self.activ = get_activation_layer(mid_activation)
            if use_conv:
                self.conv2 = conv1x1(
                    in_channels=mid_channels,
                    out_channels=channels,
                    use_bias=True)
            else:
                self.fc2 = L.Linear(
                    in_size=mid_channels,
                    out_size=channels)
            self.sigmoid = get_activation_layer(out_activation)

    def __call__(self, x):
        w = F.average_pooling_2d(x, ksize=x.shape[2:])
        if not self.use_conv:
            w = F.reshape(w, shape=(w.shape[0], -1))
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = F.expand_dims(F.expand_dims(w, axis=2), axis=3)
        x = x * w
        return x


class SABlock(Chain):
    """
    Split-Attention block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    out_channels : int
        Number of output channels.
    groups : int
        Number of channel groups (cardinality, without radix).
    radix : int
        Number of splits within a cardinal group.
    reduction : int, default 4
        Squeeze reduction value.
    min_channels : int, default 32
        Minimal number of squeezed channels.
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 out_channels,
                 groups,
                 radix,
                 reduction=4,
                 min_channels=32,
                 use_conv=True,
                 bn_eps=1e-5):
        super(SABlock, self).__init__()
        self.groups = groups
        self.radix = radix
        self.use_conv = use_conv
        in_channels = out_channels * radix
        mid_channels = max(in_channels // reduction, min_channels)

        with self.init_scope():
            if use_conv:
                self.conv1 = conv1x1(
                    in_channels=out_channels,
                    out_channels=mid_channels,
                    use_bias=True)
            else:
                self.fc1 = L.Linear(
                    in_size=out_channels,
                    out_size=mid_channels)
            self.bn = L.BatchNormalization(
                size=mid_channels,
                eps=bn_eps)
            self.activ = F.relu
            if use_conv:
                self.conv2 = conv1x1(
                    in_channels=mid_channels,
                    out_channels=in_channels,
                    use_bias=True)
            else:
                self.fc2 = L.Linear(
                    in_size=mid_channels,
                    out_size=in_channels)
            self.softmax = partial(
                F.softmax,
                axis=1)

    def __call__(self, x):
        batch, channels, height, width = x.shape
        x = F.reshape(x, shape=(batch, self.radix, channels // self.radix, height, width))
        w = F.sum(x, axis=1)
        w = F.average_pooling_2d(w, ksize=w.shape[2:])
        if not self.use_conv:
            w = F.reshape(w, shape=(w.shape[0], -1))
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.bn(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = F.reshape(w, shape=(batch, self.groups, self.radix, -1))
        w = F.swapaxes(w, axis1=1, axis2=2)
        w = self.softmax(w)
        w = F.reshape(w, shape=(batch, self.radix, -1, 1, 1))
        x = x * w
        x = F.sum(x, axis=1)
        return x


class SAConvBlock(Chain):
    """
    Split-Attention convolution block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

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
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    radix : int, default 2
        Number of splits within a cardinal group.
    reduction : int, default 4
        Squeeze reduction value.
    min_channels : int, default 32
        Minimal number of squeezed channels.
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu),
                 radix=2,
                 reduction=4,
                 min_channels=32,
                 use_conv=True):
        super(SAConvBlock, self).__init__()
        with self.init_scope():
            self.conv = ConvBlock(
                in_channels=in_channels,
                out_channels=(out_channels * radix),
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                groups=(groups * radix),
                use_bias=use_bias,
                use_bn=use_bn,
                bn_eps=bn_eps,
                activation=activation)
            self.att = SABlock(
                out_channels=out_channels,
                groups=groups,
                radix=radix,
                reduction=reduction,
                min_channels=min_channels,
                use_conv=use_conv,
                bn_eps=bn_eps)

    def __call__(self, x):
        x = self.conv(x)
        x = self.att(x)
        return x


def saconv3x3_block(in_channels,
                    out_channels,
                    stride=1,
                    pad=1,
                    **kwargs):
    """
    3x3 version of the Split-Attention convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    """
    return SAConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        **kwargs)


class PixelShuffle(Chain):
    """
    Pixel-shuffle operation from 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
    Convolutional Neural Network,' https://arxiv.org/abs/1609.05158.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    """
    def __init__(self,
                 scale_factor,
                 **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def __call__(self, x):
        f1 = self.scale_factor
        f2 = self.scale_factor

        batch, channels, height, width = x.shape

        assert (channels % f1 % f2 == 0)
        new_channels = channels // f1 // f2

        x = F.reshape(x, shape=(batch, new_channels, f1 * f2, height, width))
        x = F.reshape(x, shape=(batch, new_channels, f1, f2, height, width))
        x = F.transpose(x, axes=(0, 1, 4, 2, 5, 3))
        x = F.reshape(x, shape=(batch, new_channels, height * f1, width * f2))

        return x


class DucBlock(Chain):
    """
    Dense Upsampling Convolution (DUC) block from 'Understanding Convolution for Semantic Segmentation,'
    https://arxiv.org/abs/1702.08502.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    scale_factor : int
        Multiplier for spatial size.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor,
                 **kwargs):
        super(DucBlock, self).__init__(**kwargs)
        mid_channels = (scale_factor * scale_factor) * out_channels

        with self.init_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.pix_shuffle = PixelShuffle(scale_factor=scale_factor)

    def __call__(self, x):
        x = self.conv(x)
        x = self.pix_shuffle(x)
        return x


class SimpleSequential(Chain):
    """
    A sequential chain that can be used instead of Sequential.
    """
    def __init__(self):
        super(SimpleSequential, self).__init__()
        self.layer_names = []

    def __setattr__(self, name, value):
        super(SimpleSequential, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self.layer_names.append(name)

    def __delattr__(self, name):
        super(SimpleSequential, self).__delattr__(name)
        try:
            self.layer_names.remove(name)
        except ValueError:
            pass

    def __len__(self):
        return len(self.layer_names)

    def __call__(self, x):
        for name in self.layer_names:
            x = self[name](x)
        return x

    def el(self, index):
        return self[self.layer_names[index]]


class DualPathSequential(SimpleSequential):
    """
    A sequential container for blocks with dual inputs/outputs.
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
                 dual_path_scheme_ordinal=(lambda block, x1, x2: (block(x1), x2))):
        super(DualPathSequential, self).__init__()
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def __call__(self, x1, x2=None):
        length = len(self.layer_names)
        for i, block_name in enumerate(self.layer_names):
            block = self[block_name]
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(block, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(block, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class Concurrent(SimpleSequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    """
    def __init__(self,
                 axis=1,
                 stack=False):
        super(Concurrent, self).__init__()
        self.axis = axis
        self.stack = stack

    def __call__(self, x):
        out = []
        for name in self.layer_names:
            out.append(self[name](x))
        if self.stack:
            out = F.stack(tuple(out), axis=self.axis)
        else:
            out = F.concat(tuple(out), axis=self.axis)
        return out


class SequentialConcurrent(SimpleSequential):
    """
    A sequential container with concatenated outputs.
    Blocks will be executed in the order they are added.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    cat_input : bool, default True
        Whether to concatenate input tensor.
    """
    def __init__(self,
                 axis=1,
                 stack=False,
                 cat_input=True):
        super(SequentialConcurrent, self).__init__()
        self.axis = axis
        self.stack = stack
        self.cat_input = cat_input

    def __call__(self, x):
        out = [x] if self.cat_input else []
        for name in self.layer_names:
            x = self[name](x)
            out.append(x)
        if self.stack:
            out = F.stack(tuple(out), axis=self.axis)
        else:
            out = F.concat(tuple(out), axis=self.axis)
        return out


class ParametricSequential(SimpleSequential):
    """
    A sequential container for modules with parameters.
    Blocks will be executed in the order they are added.
    """
    def __init__(self):
        super(ParametricSequential, self).__init__()

    def __call__(self, x, **kwargs):
        for name in self.layer_names:
            x = self[name](x, **kwargs)
        return x


class ParametricConcurrent(SimpleSequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=1):
        super(ParametricConcurrent, self).__init__()
        self.axis = axis

    def __call__(self, x, **kwargs):
        out = []
        for name in self.layer_names:
            out.append(self[name](x, **kwargs))
        out = F.concat(tuple(out), axis=self.axis)
        return out


class Hourglass(Chain):
    """
    A hourglass block.

    Parameters:
    ----------
    down_seq : SimpleSequential
        Down modules as sequential.
    up_seq : SimpleSequential
        Up modules as sequential.
    skip_seq : SimpleSequential
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
                 return_first_skip=False):
        super(Hourglass, self).__init__()
        self.depth = len(down_seq)
        assert (merge_type in ["add"])
        assert (len(up_seq) == self.depth)
        assert (len(skip_seq) in (self.depth, self.depth + 1))
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.extra_skip = (len(skip_seq) == self.depth + 1)

        with self.init_scope():
            self.down_seq = down_seq
            self.up_seq = up_seq
            self.skip_seq = skip_seq

    def __call__(self, x):
        y = None
        down_outs = [x]
        for down_module_name in self.down_seq.layer_names:
            down_module = self.down_seq[down_module_name]
            x = down_module(x)
            down_outs.append(x)
        for i in range(len(down_outs)):
            if i != 0:
                y = down_outs[self.depth - i]
                skip_module = self.skip_seq.el(self.depth - i)
                y = skip_module(y)
                if (y is not None) and (self.merge_type == "add"):
                    x = x + y
            if i != len(down_outs) - 1:
                if (i == 0) and self.extra_skip:
                    skip_module = self.skip_seq.el(self.depth)
                    x = skip_module(x)
                up_module = self.up_seq.el(self.depth - 1 - i)
                x = up_module(x)
        if self.return_first_skip:
            return x, y
        else:
            return x


class SesquialteralHourglass(Chain):
    """
    A sesquialteral hourglass block.

    Parameters:
    ----------
    down1_seq : SimpleSequential
        The first down modules as sequential.
    skip1_seq : SimpleSequential
        The first skip connection modules as sequential.
    up_seq : SimpleSequential
        Up modules as sequential.
    skip2_seq : SimpleSequential
        The second skip connection modules as sequential.
    down2_seq : SimpleSequential
        The second down modules as sequential.
    merge_type : str, default 'cat'
        Type of concatenation of up and skip outputs.
    """
    def __init__(self,
                 down1_seq,
                 skip1_seq,
                 up_seq,
                 skip2_seq,
                 down2_seq,
                 merge_type="cat"):
        super(SesquialteralHourglass, self).__init__()
        assert (len(down1_seq) == len(up_seq))
        assert (len(down1_seq) == len(down2_seq))
        assert (len(skip1_seq) == len(skip2_seq))
        assert (len(down1_seq) == len(skip1_seq) - 1)
        assert (merge_type in ["cat", "add"])
        self.merge_type = merge_type
        self.depth = len(down1_seq)

        with self.init_scope():
            self.down1_seq = down1_seq
            self.skip1_seq = skip1_seq
            self.up_seq = up_seq
            self.skip2_seq = skip2_seq
            self.down2_seq = down2_seq

    def _merge(self, x, y):
        if y is not None:
            if self.merge_type == "cat":
                x = F.concat((x, y), axis=1)
            elif self.merge_type == "add":
                x = x + y
        return x

    def __call__(self, x):
        y = self.skip1_seq[self.skip1_seq.layer_names[0]](x)
        skip1_outs = [y]
        for i in range(self.depth):
            x = self.down1_seq[self.down1_seq.layer_names[i]](x)
            y = self.skip1_seq[self.skip1_seq.layer_names[i + 1]](x)
            skip1_outs.append(y)
        x = skip1_outs[self.depth]
        y = self.skip2_seq[self.skip2_seq.layer_names[0]](x)
        skip2_outs = [y]
        for i in range(self.depth):
            x = self.up_seq[self.up_seq.layer_names[i]](x)
            y = skip1_outs[self.depth - 1 - i]
            x = self._merge(x, y)
            y = self.skip2_seq[self.skip2_seq.layer_names[i + 1]](x)
            skip2_outs.append(y)
        x = self.skip2_seq[self.skip2_seq.layer_names[self.depth]](x)
        for i in range(self.depth):
            x = self.down2_seq[self.down2_seq.layer_names[i]](x)
            y = skip2_outs[self.depth - 1 - i]
            x = self._merge(x, y)
        return x


class MultiOutputSequential(SimpleSequential):
    """
    A sequential container with multiple outputs.
    Blocks will be executed in the order they are added.

    Parameters:
    ----------
    multi_output : bool, default True
        Whether to return multiple output.
    dual_output : bool, default False
        Whether to return dual output.
    return_last : bool, default True
        Whether to forcibly return last value.
    """
    def __init__(self,
                 multi_output=True,
                 dual_output=False,
                 return_last=True):
        super(MultiOutputSequential, self).__init__()
        self.multi_output = multi_output
        self.dual_output = dual_output
        self.return_last = return_last

    def __call__(self, x):
        outs = []
        for name in self.layer_names:
            block = self[name]
            x = block(x)
            if hasattr(block, "do_output") and block.do_output:
                outs.append(x)
            elif hasattr(block, "do_output2") and block.do_output2:
                assert (type(x) == tuple)
                outs.extend(x[1])
                x = x[0]
        if self.multi_output:
            return [x] + outs if self.return_last else outs
        elif self.dual_output:
            return x, outs
        else:
            return x


class ParallelConcurent(SimpleSequential):
    """
    A sequential container with multiple inputs and multiple outputs.
    Modules will be executed in the order they are added.
    """
    def __init__(self):
        super(ParallelConcurent, self).__init__()

    def __call__(self, x):
        out = []
        for name, xi in zip(self.layer_names, x):
            out.append(self[name](xi))
        return out


class Flatten(Chain):
    """
    Simple flatten block.
    """
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)


class AdaptiveAvgPool2D(Chain):
    """
    Simple adaptive average pooling block.
    """
    def __call__(self, x):
        return F.average_pooling_2d(x, ksize=x.shape[2:])


class InterpolationBlock(Chain):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    out_size : tuple of 2 int, default None
        Spatial size of the output tensor for the bilinear interpolation operation.
    up : bool, default True
        Whether to upsample or downsample.
    mode : str, default 'bilinear'
        Algorithm used for upsampling.
    align_corners : bool, default True
        Whether to align the corner pixels of the input and output tensors.
    """
    def __init__(self,
                 scale_factor,
                 out_size=None,
                 up=True,
                 mode="bilinear",
                 align_corners=True,
                 **kwargs):
        super(InterpolationBlock, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.out_size = out_size
        self.up = up
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x, size=None):
        out_size = self.calc_out_size(x) if size is None else size
        return F.resize_images(x, output_shape=out_size, mode=self.mode, align_corners=self.align_corners)

    def calc_out_size(self, x):
        if self.out_size is not None:
            return self.out_size
        if self.up:
            return tuple(s * self.scale_factor for s in x.shape[2:])
        else:
            return tuple(s // self.scale_factor for s in x.shape[2:])


class HeatmapMaxDetBlock(Chain):
    """
    Heatmap maximum detector block (for human pose estimation task).
    """
    def __init__(self,
                 **kwargs):
        super(HeatmapMaxDetBlock, self).__init__(**kwargs)

    def __call__(self, x):
        heatmap = x
        vector_dim = 2
        batch = heatmap.shape[0]
        channels = heatmap.shape[1]
        in_size = x.shape[2:]
        heatmap_vector = F.reshape(heatmap, shape=(batch, channels, -1))
        indices = F.cast(F.expand_dims(F.argmax(heatmap_vector, axis=vector_dim), axis=vector_dim), np.float32)
        scores = F.max(heatmap_vector, axis=vector_dim, keepdims=True)
        scores_mask = (scores.array > 0.0).astype(np.float32)
        pts_x = (indices.array % in_size[1]) * scores_mask
        pts_y = (indices.array // in_size[1]) * scores_mask
        pts = F.concat((pts_x, pts_y, scores), axis=vector_dim).array
        for b in range(batch):
            for k in range(channels):
                hm = heatmap[b, k, :, :].array
                px = int(pts_x[b, k])
                py = int(pts_y[b, k])
                if (0 < px < in_size[1] - 1) and (0 < py < in_size[0] - 1):
                    pts[b, k, 0] += np.sign(hm[py, px + 1] - hm[py, px - 1]) * 0.25
                    pts[b, k, 1] += np.sign(hm[py + 1, px] - hm[py - 1, px]) * 0.25
        return pts
