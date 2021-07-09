"""
    Common routines for models in Gluon.
"""

__all__ = ['round_channels', 'BreakBlock', 'get_activation_layer', 'ReLU6', 'PReLU2', 'HSigmoid', 'HSwish', 'Softmax',
           'SelectableDense', 'BatchNormExtra', 'DenseBlock', 'ConvBlock1d', 'conv1x1', 'conv3x3', 'depthwise_conv3x3',
           'ConvBlock', 'conv1x1_block', 'conv3x3_block', 'conv5x5_block', 'conv7x7_block', 'dwconv_block',
           'dwconv3x3_block', 'dwconv5x5_block', 'dwsconv3x3_block', 'PreConvBlock', 'pre_conv1x1_block',
           'pre_conv3x3_block', 'DeconvBlock', 'NormActivation', 'InterpolationBlock', 'ChannelShuffle',
           'ChannelShuffle2', 'SEBlock', 'SABlock', 'SAConvBlock', 'saconv3x3_block', 'DucBlock', 'split', 'IBN',
           'DualPathSequential', 'ParametricSequential', 'Concurrent', 'SequentialConcurrent', 'ParametricConcurrent',
           'Hourglass', 'SesquialteralHourglass', 'MultiOutputSequential', 'ParallelConcurent',
           'DualPathParallelConcurent', 'HeatmapMaxDetBlock']

import math
from inspect import isfunction
import mxnet as mx
from mxnet.gluon import nn, HybridBlock


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


class BreakBlock(HybridBlock):
    """
    Break coonnection block for hourglass.
    """
    def __init__(self, prefix=None, params=None):
        super(BreakBlock, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x):
        return None

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


class ReLU6(HybridBlock):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0.0, 6.0, name="relu6")

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


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

    def __repr__(self):
        s = '{name}(in_channels={in_channels})'
        return s.format(
            name=self.__class__.__name__,
            in_channels=self.alpha.shape[0])


class HSigmoid(HybridBlock):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


class HSwish(HybridBlock):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x * F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


class Softmax(HybridBlock):
    """
    Softmax activation function.

    Parameters:
    ----------
    axis : int, default 1
        Axis along which to do softmax.
    """
    def __init__(self, axis=1, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x):
        return x.softmax(axis=self.axis)

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or HybridBlock
        Activation function or name of activation function.

    Returns:
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
        elif activation == "hsigmoid":
            return HSigmoid()
        else:
            return nn.Activation(activation)
    else:
        assert (isinstance(activation, HybridBlock))
        return activation


class SelectableDense(HybridBlock):
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
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : str or `Initializer`
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    num_options : int, default 1
        Number of selectable options.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias=False,
                 dtype="float32",
                 weight_initializer=None,
                 bias_initializer="zeros",
                 num_options=1,
                 **kwargs):
        super(SelectableDense, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.num_options = num_options

        with self.name_scope():
            self.weight = self.params.get(
                "weight",
                shape=(num_options, out_channels, in_channels),
                init=weight_initializer,
                dtype=dtype,
                allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get(
                    "bias",
                    shape=(num_options, out_channels),
                    init=bias_initializer,
                    dtype=dtype,
                    allow_deferred_init=True)
            else:
                self.bias = None

    def hybrid_forward(self, F, x, indices, weight, bias=None):
        weight = F.take(weight, indices=indices, axis=0)
        x = x.expand_dims(axis=-1)
        x = F.batch_dot(weight, x)
        x = x.squeeze(axis=-1)
        if self.use_bias:
            bias = F.take(bias, indices=indices, axis=0)
            x += bias
        return x

    def __repr__(self):
        s = "{name}({layout}, {num_options})"
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        layout="{0} -> {1}".format(shape[1] if shape[1] else None, shape[0]),
                        num_options=self.num_options)


class BatchNormExtra(nn.BatchNorm):
    """
    Batch normalization layer with extra parameters.
    """
    def __init__(self, **kwargs):
        has_cudnn_off = ("cudnn_off" in kwargs)
        if has_cudnn_off:
            cudnn_off = kwargs["cudnn_off"]
            del kwargs["cudnn_off"]
        super(BatchNormExtra, self).__init__(**kwargs)
        if has_cudnn_off:
            self._kwargs["cudnn_off"] = cudnn_off


class DenseBlock(HybridBlock):
    """
    Standard dense block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.name_scope():
            self.fc = nn.Dense(
                units=out_channels,
                use_bias=use_bias,
                in_units=in_channels)
            if self.use_bn:
                self.bn = BatchNormExtra(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats,
                    cudnn_off=bn_cudnn_off)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class ConvBlock1d(HybridBlock):
    """
    Standard 1D convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    strides : int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(ConvBlock1d, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.name_scope():
            self.conv = nn.Conv1D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=in_channels)
            if self.use_bn:
                self.bn = BatchNormExtra(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats,
                    cudnn_off=bn_cudnn_off)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


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
                      strides=1,
                      padding=1,
                      dilation=1,
                      use_bias=False):
    """
    Depthwise convolution 3x3 layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2D(
        channels=channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=channels,
        use_bias=use_bias,
        in_channels=channels)


class ConvBlock(HybridBlock):
    """
    Standard convolution block with batch normalization and activation.

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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
                 bn_cudnn_off=False,
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
                self.bn = BatchNormExtra(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats,
                    cudnn_off=bn_cudnn_off)
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
                  padding=0,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  bn_cudnn_off=False,
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
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        bn_cudnn_off=bn_cudnn_off,
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
                  bn_cudnn_off=False,
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
        bn_cudnn_off=bn_cudnn_off,
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
                  bn_cudnn_off=False,
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
        bn_cudnn_off=bn_cudnn_off,
        activation=activation,
        **kwargs)


def conv7x7_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  use_bn=True,
                  bn_use_global_stats=False,
                  bn_cudnn_off=False,
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
        bn_cudnn_off=bn_cudnn_off,
        activation=activation,
        **kwargs)


def dwconv_block(in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
    """
    Depthwise version of the standard convolution block.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        bn_cudnn_off=bn_cudnn_off,
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
                    bn_cudnn_off=False,
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        bn_cudnn_off=bn_cudnn_off,
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
                    bn_cudnn_off=False,
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        bn_cudnn_off=bn_cudnn_off,
        activation=activation,
        **kwargs)


class DwsConvBlock(HybridBlock):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    dw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the pointwise convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 dw_activation=(lambda: nn.Activation("relu")),
                 pw_activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.dw_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                use_bias=use_bias,
                use_bn=dw_use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=dw_activation)
            self.pw_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=pw_use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=pw_activation)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=1,
                     dilation=1,
                     use_bias=False,
                     bn_epsilon=1e-5,
                     bn_use_global_stats=False,
                     bn_cudnn_off=False,
                     dw_activation=(lambda: nn.Activation("relu")),
                     pw_activation=(lambda: nn.Activation("relu")),
                     **kwargs):
    """
    3x3 depthwise separable version of the standard convolution block.

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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    dw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the pointwise convolution block.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        bn_cudnn_off=bn_cudnn_off,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
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
                 use_bn=True,
                 bn_use_global_stats=False,
                 return_preact=False,
                 activate=True,
                 **kwargs):
        super(PreConvBlock, self).__init__(**kwargs)
        self.return_preact = return_preact
        self.activate = activate
        self.use_bn = use_bn

        with self.name_scope():
            if self.use_bn:
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
                      strides=1,
                      use_bias=False,
                      use_bn=True,
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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
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
        use_bn=use_bn,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        activate=activate)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      use_bias=False,
                      use_bn=True,
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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
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
        use_bias=use_bias,
        use_bn=use_bn,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        activate=activate)


class DeconvBlock(HybridBlock):
    """
    Deconvolution block with batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the deconvolution.
    padding : int or tuple/list of 2 int
        Padding value for deconvolution layer.
    out_padding : int or tuple/list of 2 int
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for deconvolution layer.
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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 out_padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.name_scope():
            self.conv = nn.Conv2DTranspose(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                output_padding=out_padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=in_channels)
            if self.use_bn:
                self.bn = BatchNormExtra(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats,
                    cudnn_off=bn_cudnn_off)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class NormActivation(HybridBlock):
    """
    Activation block with preliminary batch normalization. It's used by itself as the final block in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(NormActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = BatchNormExtra(
                in_channels=in_channels,
                epsilon=bn_epsilon,
                use_global_stats=bn_use_global_stats,
                cudnn_off=bn_cudnn_off)
            self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class InterpolationBlock(HybridBlock):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    out_size : tuple of 2 int, default None
        Spatial size of the output tensor for the bilinear interpolation operation.
    bilinear : bool, default True
        Whether to use bilinear interpolation.
    up : bool, default True
        Whether to upsample or downsample.
    """
    def __init__(self,
                 scale_factor,
                 out_size=None,
                 bilinear=True,
                 up=True,
                 **kwargs):
        super(InterpolationBlock, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.out_size = out_size
        self.bilinear = bilinear
        self.up = up

    def hybrid_forward(self, F, x, size=None):
        if self.bilinear or (size is not None):
            out_size = self.calc_out_size(x) if size is None else size
            return F.contrib.BilinearResize2D(x, height=out_size[0], width=out_size[1])
        else:
            return F.UpSampling(x, scale=self.scale_factor, sample_type="nearest")

    def calc_out_size(self, x):
        if self.out_size is not None:
            return self.out_size
        if self.up:
            return tuple(s * self.scale_factor for s in x.shape[2:])
        else:
            return tuple(s // self.scale_factor for s in x.shape[2:])

    def __repr__(self):
        s = '{name}(scale_factor={scale_factor}, out_size={out_size}, bilinear={bilinear}, up={up})'
        return s.format(
            name=self.__class__.__name__,
            scale_factor=self.scale_factor,
            out_size=self.out_size,
            bilinear=self.bilinear,
            up=self.up)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        if self.bilinear:
            num_flops = 9 * x.size
        else:
            num_flops = 4 * x.size
        num_macs = 0
        return num_flops, num_macs


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

    Returns:
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

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(
            name=self.__class__.__name__,
            groups=self.groups)


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

    Returns:
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
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function, or str, or HybridBlock, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or HybridBlock, default 'sigmoid'
        Activation function after the last convolution.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 mid_channels=None,
                 round_mid=False,
                 use_conv=True,
                 mid_activation=(lambda: nn.Activation("relu")),
                 out_activation=(lambda: nn.Activation("sigmoid")),
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        with self.name_scope():
            if use_conv:
                self.conv1 = conv1x1(
                    in_channels=channels,
                    out_channels=mid_channels,
                    use_bias=True)
            else:
                self.fc1 = nn.Dense(
                    in_units=channels,
                    units=mid_channels)
            self.activ = get_activation_layer(mid_activation)
            if use_conv:
                self.conv2 = conv1x1(
                    in_channels=mid_channels,
                    out_channels=channels,
                    use_bias=True)
            else:
                self.fc2 = nn.Dense(
                    in_units=mid_channels,
                    units=channels)
            self.sigmoid = get_activation_layer(out_activation)

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        if not self.use_conv:
            w = F.Flatten(w)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.expand_dims(2).expand_dims(3)
        x = F.broadcast_mul(x, w)
        return x


class SABlock(HybridBlock):
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
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 out_channels,
                 groups,
                 radix,
                 reduction=4,
                 min_channels=32,
                 use_conv=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(SABlock, self).__init__(**kwargs)
        self.groups = groups
        self.radix = radix
        self.use_conv = use_conv
        in_channels = out_channels * radix
        mid_channels = max(in_channels // reduction, min_channels)

        with self.name_scope():
            if use_conv:
                self.conv1 = conv1x1(
                    in_channels=out_channels,
                    out_channels=mid_channels,
                    use_bias=True)
            else:
                self.fc1 = nn.Dense(
                    in_units=out_channels,
                    units=mid_channels)
            self.bn = BatchNormExtra(
                in_channels=mid_channels,
                epsilon=bn_epsilon,
                use_global_stats=bn_use_global_stats,
                cudnn_off=bn_cudnn_off)
            self.activ = nn.Activation("relu")
            if use_conv:
                self.conv2 = conv1x1(
                    in_channels=mid_channels,
                    out_channels=in_channels,
                    use_bias=True)
            else:
                self.fc2 = nn.Dense(
                    in_units=mid_channels,
                    units=in_channels)

    def hybrid_forward(self, F, x):
        x = x.reshape((0, -4, self.radix, -1, -2))
        w = x.sum(axis=1)
        w = F.contrib.AdaptiveAvgPooling2D(w, output_size=1)
        if not self.use_conv:
            w = F.Flatten(w)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.bn(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = w.reshape((0, self.groups, self.radix, -1))
        w = w.swapaxes(1, 2)
        w = F.softmax(w, axis=1)
        w = w.reshape((0, self.radix, -1, 1, 1))
        x = F.broadcast_mul(x, w)
        x = x.sum(axis=1)
        return x


class SAConvBlock(HybridBlock):
    """
    Split-Attention convolution block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
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
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 radix=2,
                 reduction=4,
                 min_channels=32,
                 use_conv=True,
                 **kwargs):
        super(SAConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = ConvBlock(
                in_channels=in_channels,
                out_channels=(out_channels * radix),
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=(groups * radix),
                use_bias=use_bias,
                use_bn=use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=activation)
            self.att = SABlock(
                out_channels=out_channels,
                groups=groups,
                radix=radix,
                reduction=reduction,
                min_channels=min_channels,
                use_conv=use_conv,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.att(x)
        return x


def saconv3x3_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=1,
                    **kwargs):
    """
    3x3 version of the Split-Attention convolution block.

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
    """
    return SAConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        **kwargs)


class PixelShuffle(HybridBlock):
    """
    Pixel-shuffle operation from 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
    Convolutional Neural Network,' https://arxiv.org/abs/1609.05158.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    in_size : tuple of 2 int
        Spatial size of the input heatmap tensor.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    """
    def __init__(self,
                 channels,
                 scale_factor,
                 in_size,
                 fixed_size,
                 **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        assert (channels % scale_factor % scale_factor == 0)
        self.channels = channels
        self.scale_factor = scale_factor
        self.in_size = in_size
        self.fixed_size = fixed_size

    def hybrid_forward(self, F, x):
        f1 = self.scale_factor
        f2 = self.scale_factor

        if not self.fixed_size:
            x = x.reshape((0, -4, -1, f1 * f2, 0, 0))
            x = x.reshape((0, 0, -4, f1, f2, 0, 0))
            x = x.transpose((0, 1, 4, 2, 5, 3))
            x = x.reshape((0, 0, -3, -3))
        else:
            new_channels = self.channels // f1 // f2
            h, w = self.in_size
            x = x.reshape((0, new_channels, f1 * f2, h, w))
            x = x.reshape((0, new_channels, f1, f2, h, w))
            x = x.transpose((0, 1, 4, 2, 5, 3))
            x = x.reshape((0, new_channels, h * f1, w * f2))
        return x


class DucBlock(HybridBlock):
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
    in_size : tuple of 2 int
        Spatial size of the input heatmap tensor.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor,
                 in_size,
                 fixed_size,
                 bn_use_global_stats,
                 bn_cudnn_off,
                 **kwargs):
        super(DucBlock, self).__init__(**kwargs)
        mid_channels = (scale_factor * scale_factor) * out_channels

        with self.name_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.pix_shuffle = PixelShuffle(
                channels=mid_channels,
                scale_factor=scale_factor,
                in_size=in_size,
                fixed_size=fixed_size)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pix_shuffle(x)
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

    Returns:
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
    merge_type : str, default None
        Type of branch merging.
    branches : list of HybridBlock, default None
        Whether to concatenate tensors along a new dimension.
    """
    def __init__(self,
                 axis=1,
                 stack=False,
                 merge_type=None,
                 branches=None,
                 **kwargs):
        super(Concurrent, self).__init__(**kwargs)
        assert (merge_type is None) or (merge_type in ["cat", "stack", "sum"])
        self.axis = axis
        self.stack = stack
        if merge_type is not None:
            self.merge_type = merge_type
        else:
            self.merge_type = "stack" if stack else "cat"
        if branches is not None:
            with self.name_scope():
                for branch in branches:
                    self.add(branch)

    def hybrid_forward(self, F, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        if self.merge_type == "stack":
            out = F.stack(*out, axis=self.axis)
        elif self.merge_type == "cat":
            out = F.concat(*out, dim=self.axis)
        elif self.merge_type == "sum":
            out = F.stack(*out, axis=self.axis).sum(axis=self.axis)
        else:
            raise NotImplementedError()
        return out


class SequentialConcurrent(nn.HybridSequential):
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
                 cat_input=True,
                 **kwargs):
        super(SequentialConcurrent, self).__init__(**kwargs)
        self.axis = axis
        self.stack = stack
        self.cat_input = cat_input

    def hybrid_forward(self, F, x):
        out = [x] if self.cat_input else []
        for block in self._children.values():
            x = block(x)
            out.append(x)
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
        self.depth = len(down_seq)
        assert (merge_type in ["cat", "add"])
        assert (len(up_seq) == self.depth)
        assert (len(skip_seq) in (self.depth, self.depth + 1))
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.extra_skip = (len(skip_seq) == self.depth + 1)

        with self.name_scope():
            self.down_seq = down_seq
            self.up_seq = up_seq
            self.skip_seq = skip_seq

    def _merge(self, F, x, y):
        if y is not None:
            if self.merge_type == "cat":
                x = F.concat(x, y, dim=1)
            elif self.merge_type == "add":
                x = x + y
        return x

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
                x = self._merge(F, x, y)
            if i != len(down_outs) - 1:
                if (i == 0) and self.extra_skip:
                    skip_module = self.skip_seq[self.depth]
                    x = skip_module(x)
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
    merge_type : str, default 'cat'
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
                 return_last=True,
                 **kwargs):
        super(MultiOutputSequential, self).__init__(**kwargs)
        self.multi_output = multi_output
        self.dual_output = dual_output
        self.return_last = return_last

    def hybrid_forward(self, F, x):
        outs = []
        for block in self._children.values():
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


class ParallelConcurent(nn.HybridSequential):
    """
    A sequential container with multiple inputs and multiple outputs.
    Modules will be executed in the order they are added.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    merge_type : str, default 'list'
        Type of branch merging.
    """
    def __init__(self,
                 axis=1,
                 merge_type="list",
                 **kwargs):
        super(ParallelConcurent, self).__init__(**kwargs)
        assert (merge_type is None) or (merge_type in ["list", "cat", "stack", "sum"])
        self.axis = axis
        self.merge_type = merge_type

    def hybrid_forward(self, F, x):
        out = []
        for block, xi in zip(self._children.values(), x):
            out.append(block(xi))
        if self.merge_type == "list":
            pass
        elif self.merge_type == "stack":
            out = F.stack(*out, axis=self.axis)
        elif self.merge_type == "cat":
            out = F.concat(*out, dim=self.axis)
        elif self.merge_type == "sum":
            out = F.stack(*out, axis=self.axis).sum(axis=self.axis)
        else:
            raise NotImplementedError()
        return out


class DualPathParallelConcurent(nn.HybridSequential):
    """
    A sequential container with multiple dual-path inputs and single/multiple outputs.
    Blocks will be executed in the order they are added.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    merge_type : str, default 'list'
        Type of branch merging.
    """
    def __init__(self,
                 axis=1,
                 merge_type="list",
                 **kwargs):
        super(DualPathParallelConcurent, self).__init__(**kwargs)
        assert (merge_type is None) or (merge_type in ["list", "cat", "stack", "sum"])
        self.axis = axis
        self.merge_type = merge_type

    def hybrid_forward(self, F, x1, x2):
        x1_out = []
        x2_out = []
        for block, x1i, x2i in zip(self._children.values(), x1, x2):
            y1i, y2i = block(x1i, x2i)
            x1_out.append(y1i)
            x2_out.append(y2i)
        if self.merge_type == "list":
            pass
        elif self.merge_type == "stack":
            x1_out = F.stack(*x1_out, axis=self.axis)
            x2_out = F.stack(*x2_out, axis=self.axis)
        elif self.merge_type == "cat":
            x1_out = F.concat(*x1_out, dim=self.axis)
            x2_out = F.concat(*x2_out, dim=self.axis)
        elif self.merge_type == "sum":
            x1_out = F.stack(*x1_out, axis=self.axis).sum(axis=self.axis)
            x2_out = F.stack(*x2_out, axis=self.axis).sum(axis=self.axis)
        else:
            raise NotImplementedError()
        return x1_out, x2_out


class HeatmapMaxDetBlock(HybridBlock):
    """
    Heatmap maximum detector block (for human pose estimation task).

    Parameters:
    ----------
    channels : int
        Number of channels.
    in_size : tuple of 2 int
        Spatial size of the input heatmap tensor.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    tune : bool, default True
        Whether to tune point positions.
    """
    def __init__(self,
                 channels,
                 in_size,
                 fixed_size,
                 tune=True,
                 **kwargs):
        super(HeatmapMaxDetBlock, self).__init__(**kwargs)
        self.channels = channels
        self.in_size = in_size
        self.fixed_size = fixed_size
        self.tune = tune

    def hybrid_forward(self, F, x):
        # assert (not self.fixed_size) or (self.in_size == x.shape[2:])
        vector_dim = 2
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        heatmap_vector = x.reshape((0, 0, -3))
        indices = heatmap_vector.argmax(axis=vector_dim, keepdims=True)
        scores = heatmap_vector.max(axis=vector_dim, keepdims=True)
        scores_mask = (scores > 0.0)
        pts_x = (indices % in_size[1]) * scores_mask
        pts_y = (indices / in_size[1]).floor() * scores_mask
        pts = F.concat(pts_x, pts_y, scores, dim=vector_dim)
        if self.tune:
            batch = x.shape[0]
            for b in range(batch):
                for k in range(self.channels):
                    hm = x[b, k, :, :]
                    px = int(pts[b, k, 0].asscalar())
                    py = int(pts[b, k, 1].asscalar())
                    if (0 < px < in_size[1] - 1) and (0 < py < in_size[0] - 1):
                        pts[b, k, 0] += (hm[py, px + 1] - hm[py, px - 1]).sign() * 0.25
                        pts[b, k, 1] += (hm[py + 1, px] - hm[py - 1, px]).sign() * 0.25
        return pts

    def __repr__(self):
        s = "{name}(channels={channels}, in_size={in_size}, fixed_size={fixed_size})"
        return s.format(
            name=self.__class__.__name__,
            channels=self.channels,
            in_size=self.in_size,
            fixed_size=self.fixed_size)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        num_flops = x.size + 26 * self.channels
        num_macs = 0
        return num_flops, num_macs
