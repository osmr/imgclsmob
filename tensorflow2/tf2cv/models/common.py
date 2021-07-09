"""
    Common routines for models in TensorFlow 2.0.
"""

__all__ = ['is_channels_first', 'get_channel_axis', 'round_channels', 'get_im_size', 'interpolate_im', 'BreakBlock',
           'ReLU6', 'HSwish', 'PReLU2', 'get_activation_layer', 'flatten', 'MaxPool2d', 'AvgPool2d', 'GlobalAvgPool2d',
           'BatchNorm', 'InstanceNorm', 'IBN', 'Conv1d', 'Conv2d', 'SelectableDense', 'DenseBlock', 'ConvBlock1d',
           'conv1x1', 'conv3x3', 'depthwise_conv3x3', 'ConvBlock', 'conv1x1_block', 'conv3x3_block', 'conv5x5_block',
           'conv7x7_block', 'dwconv_block', 'dwconv3x3_block', 'dwconv5x5_block', 'dwsconv3x3_block', 'PreConvBlock',
           'pre_conv1x1_block', 'pre_conv3x3_block', 'DeconvBlock', 'ChannelShuffle', 'ChannelShuffle2', 'SEBlock',
           'SABlock', 'SAConvBlock', 'saconv3x3_block', 'PixelShuffle', 'DucBlock', 'Identity', 'SimpleSequential',
           'ParametricSequential', 'DualPathSequential', 'Concurrent', 'SequentialConcurrent', 'ParametricConcurrent',
           'MultiOutputSequential', 'ParallelConcurent', 'DualPathParallelConcurent', 'NormActivation',
           'InterpolationBlock', 'Hourglass', 'HeatmapMaxDetBlock']

import math
from inspect import isfunction
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec


def is_channels_first(data_format):
    """
    Is tested data format channels first.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns:
    -------
    bool
        A flag.
    """
    return data_format == "channels_first"


def get_channel_axis(data_format):
    """
    Get channel axis.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns:
    -------
    int
        Channel axis.
    """
    return 1 if is_channels_first(data_format) else -1


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


def get_im_size(x,
                data_format):
    """
    Get spatial size for a tensor.

    Parameters:
    ----------
    x : tensor
        A tensor.
    data_format : str
        The ordering of the dimensions in the tensor.

    Returns:
    -------
    (int, int)
        Size (height x width).
    """
    x_shape = x.get_shape().as_list()
    return x_shape[2:4] if is_channels_first(data_format) else x_shape[1:3]


def interpolate_im(x,
                   scale_factor=1,
                   out_size=None,
                   data_format="channels_last"):
    """
    Bilinear change spatial size for a tensor.

    Parameters:
    ----------
    x : tensor
        A tensor.
    scale_factor : int, default 1
        Multiplier for spatial size.
    out_size : tuple of 2 int, default None
        Spatial size of the output tensor for the bilinear upsampling operation.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns:
    -------
    tensor
        Resulted tensor.
    """
    if out_size is None:
        in_size = get_im_size(x, data_format=data_format)
        out_size = tuple(i * scale_factor for i in in_size) if scale_factor != 0 else in_size
    if is_channels_first(data_format):
        x = tf.transpose(x, perm=[0, 2, 3, 1])
    x = tf.image.resize(
        images=x,
        size=out_size)
    if is_channels_first(data_format):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
    return x


class BreakBlock(nn.Layer):
    """
    Break coonnection block for hourglass.
    """
    def __init__(self, **kwargs):
        super(BreakBlock, self).__init__(**kwargs)

    def call(self, x):
        return None

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


class ReLU6(nn.Layer):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x)


class Swish(nn.Layer):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def call(self, x):
        return x * tf.nn.sigmoid(x)


class HSigmoid(nn.Layer):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x + 3.0) / 6.0


class HSwish(nn.Layer):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


class PReLU2(nn.PReLU):
    """
    Parametric leaky version of a Rectified Linear Unit (with wide alpha).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    alpha_initializer : tf.Initializer, default tf.constant_initializer(0.25)
        Initializer function for the weights.
    shared_axes : list of int, default None
        The axes along which to share learnable parameters for the activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels=1,
                 alpha_initializer=tf.constant_initializer(0.25),
                 data_format="channels_last",
                 **kwargs):
        self.in_channels = in_channels
        self.data_format = data_format
        super(PReLU2, self).__init__(
            alpha_initializer=alpha_initializer,
            **kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(self.in_channels,),
            name="alpha",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint)
        channel_axis = (1 if is_channels_first(self.data_format) else len(input_shape) - 1)
        assert (self.in_channels == input_shape[channel_axis])
        axes = {}
        for i in range(1, len(input_shape)):
            if i != channel_axis:
                axes[i] = input_shape[i]
        from tensorflow.python.keras.engine.input_spec import InputSpec
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, x):
        if is_channels_first(self.data_format) and (len(x.shape.as_list()) == 4):
            x = tf.transpose(x, perm=[0, 2, 3, 1])

        pos = K.relu(x)
        neg = -self.alpha * K.relu(-x)
        x = pos + neg

        if is_channels_first(self.data_format) and (len(x.shape.as_list()) == 4):
            x = tf.transpose(x, perm=[0, 3, 1, 2])
        return x


class Tanh(nn.Layer):
    """
    Tanh activation function.
    """
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def call(self, x):
        return tf.math.tanh(x)


def get_activation_layer(activation,
                         **kwargs):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Layer
        Activation function or name of activation function.

    Returns:
    -------
    nn.Layer
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(**kwargs)
        elif activation == "relu6":
            return ReLU6(**kwargs)
        elif activation == "prelu2":
            return PReLU2(**kwargs)
        elif activation == "swish":
            return Swish(**kwargs)
        elif activation == "hswish":
            return HSwish(**kwargs)
        elif activation == "sigmoid":
            return tf.nn.sigmoid
        elif activation == "hsigmoid":
            return HSigmoid(**kwargs)
        elif activation == "tanh":
            return Tanh(**kwargs)
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Layer))
        return activation


def flatten(x,
            data_format):
    """
    Flattens the input to two dimensional.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns:
    -------
    Tensor
        Resulted tensor.
    """
    if not is_channels_first(data_format):
        x = tf.transpose(x, perm=(0, 3, 1, 2))
    x = tf.reshape(x, shape=(-1, np.prod(x.get_shape().as_list()[1:])))
    return x


class MaxPool2d(nn.Layer):
    """
    Max pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 pool_size,
                 strides,
                 padding=0,
                 ceil_mode=False,
                 data_format="channels_last",
                 **kwargs):
        super(MaxPool2d, self).__init__(**kwargs)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.use_stride = (strides[0] > 1) or (strides[1] > 1)
        self.ceil_mode = ceil_mode and self.use_stride
        self.use_pad = (padding[0] > 0) or (padding[1] > 0)

        if self.ceil_mode:
            self.padding = padding
            self.pool_size = pool_size
            self.strides = strides
            self.data_format = data_format
        elif self.use_pad:
            if is_channels_first(data_format):
                self.paddings_tf = [[0, 0], [0, 0], [padding[0]] * 2, [padding[1]] * 2]
            else:
                self.paddings_tf = [[0, 0], [padding[0]] * 2, [padding[1]] * 2, [0, 0]]

        self.pool = nn.MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            padding="valid",
            data_format=data_format)

    def call(self, x):
        if self.ceil_mode:
            x_shape = x.get_shape().as_list()
            if is_channels_first(self.data_format):
                height = x_shape[2]
                width = x_shape[3]
            else:
                height = x_shape[1]
                width = x_shape[2]
            padding = self.padding
            out_height = float(height + 2 * padding[0] - self.pool_size[0]) / self.strides[0] + 1.0
            out_width = float(width + 2 * padding[1] - self.pool_size[1]) / self.strides[1] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)
            if (padding[0] > 0) or (padding[1] > 0):
                if is_channels_first(self.data_format):
                    paddings_tf = [[0, 0], [0, 0], [padding[0]] * 2, [padding[1]] * 2]
                else:
                    paddings_tf = [[0, 0], [padding[0]] * 2, [padding[1]] * 2, [0, 0]]
                x = tf.pad(x, paddings=paddings_tf)
        elif self.use_pad:
            x = tf.pad(x, paddings=self.paddings_tf)

        x = self.pool(x)
        return x


class AvgPool2d(nn.Layer):
    """
    Average pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 pool_size,
                 strides,
                 padding=0,
                 ceil_mode=False,
                 data_format="channels_last",
                 **kwargs):
        super(AvgPool2d, self).__init__(**kwargs)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.use_stride = (strides[0] > 1) or (strides[1] > 1)
        self.ceil_mode = ceil_mode and self.use_stride
        self.use_pad = (padding[0] > 0) or (padding[1] > 0)

        if self.ceil_mode:
            self.padding = padding
            self.pool_size = pool_size
            self.strides = strides
            self.data_format = data_format
        elif self.use_pad:
            if is_channels_first(data_format):
                self.paddings_tf = [[0, 0], [0, 0], [padding[0]] * 2, [padding[1]] * 2]
            else:
                self.paddings_tf = [[0, 0], [padding[0]] * 2, [padding[1]] * 2, [0, 0]]

        self.pool = nn.AveragePooling2D(
            pool_size=pool_size,
            strides=1,
            padding="valid",
            data_format=data_format,
            name="pool")
        if self.use_stride:
            self.stride_pool = nn.AveragePooling2D(
                pool_size=1,
                strides=strides,
                padding="valid",
                data_format=data_format,
                name="stride_pool")

    def call(self, x, training=None):
        if self.ceil_mode:
            x_shape = x.get_shape().as_list()
            if is_channels_first(self.data_format):
                height = x_shape[2]
                width = x_shape[3]
            else:
                height = x_shape[1]
                width = x_shape[2]
            padding = self.padding
            out_height = float(height + 2 * padding[0] - self.pool_size[0]) / self.strides[0] + 1.0
            out_width = float(width + 2 * padding[1] - self.pool_size[1]) / self.strides[1] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)
            if (padding[0] > 0) or (padding[1] > 0):
                if is_channels_first(self.data_format):
                    paddings_tf = [[0, 0], [0, 0], [padding[0]] * 2, [padding[1]] * 2]
                else:
                    paddings_tf = [[0, 0], [padding[0]] * 2, [padding[1]] * 2, [0, 0]]
                x = tf.pad(x, paddings=paddings_tf)
        elif self.use_pad:
            x = tf.pad(x, paddings=self.paddings_tf)

        x = self.pool(x)
        if self.use_stride:
            x = self.stride_pool(x)
        return x


class GlobalAvgPool2d(nn.GlobalAvgPool2D):
    """
    Global average pooling.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(GlobalAvgPool2d, self).__init__(data_format=data_format, **kwargs)
        self.axis = get_channel_axis(data_format)

    def call(self, x, training=None):
        x = super(GlobalAvgPool2d, self).call(x, training)
        x = tf.expand_dims(tf.expand_dims(x, axis=self.axis), axis=self.axis)
        return x


class BatchNorm(nn.BatchNormalization):
    """
    MXNet/Gluon-like batch normalization.

    Parameters:
    ----------
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(BatchNorm, self).__init__(
            axis=get_channel_axis(data_format),
            momentum=momentum,
            epsilon=epsilon,
            **kwargs)


class InstanceNorm(nn.Layer):
    """
    MXNet/Gluon-like instance normalization layer as in 'Instance Normalization: The Missing Ingredient for Fast
    Stylization' (https://arxiv.org/abs/1607.08022). On the base of `tensorflow_addons` implementation.

    Parameters:
    ----------
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center : bool, default True
        If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored.
    scale : bool, default False
        If True, multiply by `gamma`. If False, `gamma` is not used.
    beta_initializer : str, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer : str, default 'ones'
        Initializer for the gamma weight.
    beta_regularizer : object or None, default None
        Optional regularizer for the beta weight.
    gamma_regularizer : object or None, default None
        Optional regularizer for the gamma weight.
    beta_constraint : object or None, default None
        Optional constraint for the beta weight.
    gamma_constraint : object or None, default None
        Optional constraint for the gamma weight.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 epsilon=1e-5,
                 center=True,
                 scale=False,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 data_format="channels_last",
                 **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = -1
        self.axis = get_channel_axis(data_format)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):
        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super(InstanceNorm, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)
        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)
        outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)
        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon)
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError("Axis {} of input tensor should have a defined dimension but the layer received an input "
                             "with shape {}".format(self.axis, input_shape))

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]
        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):
        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError("Number of groups ({}) cannot be more than the number of channels ({})".format(
                self.groups, dim))
        if (dim % self.groups) != 0:
            raise ValueError('Number of groups ({}) must be a multiple of the number of channels ({})'.format(
                self.groups, dim))

    def _check_axis(self):
        if self.axis == 0:
            raise ValueError("You are trying to normalize your batch axis. Do you want to use "
                             "tf.layer.batch_normalization instead")

    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape),
            axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape


class IBN(nn.Layer):
    """
    Instance-Batch Normalization block from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    channels : int
        Number of channels.
    inst_fraction : float, default 0.5
        The first fraction of channels for normalization.
    inst_first : bool, default True
        Whether instance normalization be on the first part of channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 first_fraction=0.5,
                 inst_first=True,
                 data_format="channels_last",
                 **kwargs):
        super(IBN, self).__init__(**kwargs)
        self.inst_first = inst_first
        self.data_format = data_format
        h1_channels = int(math.floor(channels * first_fraction))
        h2_channels = channels - h1_channels
        self.split_sections = [h1_channels, h2_channels]

        if self.inst_first:
            self.inst_norm = InstanceNorm(
                scale=True,
                data_format=data_format,
                name="inst_norm")
            self.batch_norm = BatchNorm(
                data_format=data_format,
                name="batch_norm")
        else:
            self.batch_norm = BatchNorm(
                data_format=data_format,
                name="batch_norm")
            self.inst_norm = InstanceNorm(
                scale=True,
                data_format=data_format,
                name="inst_norm")

    def call(self, x, training=None):
        axis = get_channel_axis(self.data_format)
        x1, x2 = tf.split(x, num_or_size_splits=self.split_sections, axis=axis)
        if self.inst_first:
            x1 = self.inst_norm(x1, training=training)
            x2 = self.batch_norm(x2, training=training)
        else:
            x1 = self.batch_norm(x1, training=training)
            x2 = self.inst_norm(x2, training=training)
        x = tf.concat([x1, x2], axis=axis)
        return x


class Conv1d(nn.Layer):
    """
    Standard 1D convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    strides : int, default 1
        Strides of the convolution.
    padding : int, default 0
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    force_same : bool, default False
        Whether to forcibly set `same` padding.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 force_same=False,
                 data_format="channels_last",
                 **kwargs):
        super(Conv1d, self).__init__(**kwargs)
        assert (in_channels is not None)
        assert (not force_same) or ((padding == kernel_size // 2) and (strides == 1) and (dilation == 1))
        self.use_conv = (groups == 1)
        self.use_dw_conv = (groups > 1) and (groups == out_channels) and (out_channels == in_channels)
        self.data_format = data_format

        self.use_pad = (padding > 0) and (not force_same)
        if self.use_pad:
            self.pad = nn.ZeroPadding1D(padding=padding)

        if self.use_conv:
            self.conv = nn.Conv1D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=("valid" if not force_same else "same"),
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                name="conv")
        elif self.use_dw_conv:
            self.dw_conv = nn.DepthwiseConv2D(
                kernel_size=(kernel_size, 1),
                strides=strides,
                padding=("valid" if not force_same else "same"),
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                name="dw_conv")

    def call(self, x):
        if self.use_pad:
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=(0, 2, 1))
            x = self.pad(x)
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=(0, 2, 1))
        if self.use_conv:
            x = self.conv(x)
        elif self.use_dw_conv:
            if is_channels_first(self.data_format):
                x = tf.expand_dims(x, axis=3)
            else:
                x = tf.expand_dims(x, axis=2)
            x = self.dw_conv(x)
            if is_channels_first(self.data_format):
                x = tf.squeeze(x, axis=3)
            else:
                x = tf.squeeze(x, axis=2)
        return x


class Conv2d(nn.Layer):
    """
    Standard convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    force_same : bool, default False
        Whether to forcibly set `same` padding.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 force_same=False,
                 data_format="channels_last",
                 **kwargs):
        super(Conv2d, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.data_format = data_format
        self.use_conv = (groups == 1)
        self.use_dw_conv = (groups > 1) and (groups == out_channels) and (out_channels == in_channels)

        # assert (strides == 1) or (dilation == 1)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.use_pad = ((padding[0] > 0) or (padding[1] > 0)) and (not force_same)
        if self.use_pad:
            self.pad = nn.ZeroPadding2D(
                padding=padding,
                data_format=data_format)
            # if is_channels_first(data_format):
            #     self.paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
            # else:
            #     self.paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]

        # self.use_post_pad = (dilation[0] > 1) and (dilation[0] % 2 == 1) and (dilation[0] == dilation[1]) and\
        #                     (dilation[0] == padding[1]) and (padding[0] == padding[1])
        # if self.use_post_pad:
        #     self.post_pad = nn.ZeroPadding2D(
        #         padding=((1, 0), (1, 0)),
        #         data_format=data_format)

        assert (not force_same) or ((padding[0] == kernel_size[0] // 2) and (padding[1] == kernel_size[1] // 2) and
                                    (strides[0] == 1) and (strides[0] == strides[1]) and (dilation[0] == 1) and
                                    (dilation[0] == dilation[1]))

        if self.use_conv:
            self.conv = nn.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=("valid" if not force_same else "same"),
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                name="conv")
        elif self.use_dw_conv:
            # assert (dilation[0] == 1) and (dilation[1] == 1)
            self.dw_conv = nn.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding=("valid" if not force_same else "same"),
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                name="dw_conv")
        else:
            assert (groups > 1)
            assert (in_channels % groups == 0)
            assert (out_channels % groups == 0)
            self.groups = groups
            self.convs = []
            for i in range(groups):
                self.convs.append(nn.Conv2D(
                    filters=(out_channels // groups),
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    data_format=data_format,
                    dilation_rate=dilation,
                    use_bias=use_bias,
                    name="convgroup{}".format(i + 1)))

    def call(self, x):
        if self.use_pad:
            x = self.pad(x)
            # x = tf.pad(x, paddings=self.paddings_tf)
        if self.use_conv:
            try:
                x = self.conv(x)
            except tf.errors.InvalidArgumentError as ex:
                if self.conv.dilation_rate != (1, 1):
                    conv_ = nn.Conv2D(
                        filters=self.conv.filters,
                        kernel_size=self.conv.kernel_size,
                        strides=self.conv.strides,
                        padding="valid",
                        data_format=self.data_format,
                        dilation_rate=self.conv.dilation_rate,
                        use_bias=self.conv.use_bias,
                        name="conv_")
                    _ = conv_(x)
                    conv_.weights[0].assign(self.conv.weights[0])
                    if len(self.conv.weights) > 1:
                        conv_.weights[1].assign(self.conv.weights[1])
                    x = conv_(x)
                else:
                    raise ex
            # x = self.conv(x)
        elif self.use_dw_conv:
            x = self.dw_conv(x)
        else:
            yy = []
            xx = tf.split(x, num_or_size_splits=self.groups, axis=get_channel_axis(self.data_format))
            for xi, convi in zip(xx, self.convs):
                yy.append(convi(xi))
            x = tf.concat(yy, axis=get_channel_axis(self.data_format))
        # if self.use_post_pad:
        #     x = self.post_pad(x)
        return x


class SelectableDense(nn.Layer):
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
    weight_initializer : str or `Initializer`, default 'glorot_uniform'
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
                 weight_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 num_options=1,
                 **kwargs):
        super(SelectableDense, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.num_options = num_options

        self.weight_initializer = initializers.get(weight_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.weight = self.add_weight(
            "weight",
            shape=[self.num_options, self.out_channels, self.in_channels],
            initializer=self.weight_initializer,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.num_options, self.out_channels],
                initializer=self.bias_initializer,
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, x, indices):
        weight = tf.gather(self.weight.value(), indices=indices, axis=0)
        x = tf.expand_dims(x, axis=-1)
        x = tf.keras.backend.batch_dot(weight, x)
        x = tf.squeeze(x, axis=-1)
        if self.use_bias:
            bias = tf.gather(self.bias.value(), indices=indices, axis=0)
            x += bias
        return x

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.out_channels)

    def get_config(self):
        config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bias": self.use_bias,
            "num_options": self.num_options,
            "weight_initializer": initializers.serialize(self.weight_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
        }
        base_config = super(SelectableDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseBlock(nn.Layer):
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.fc = nn.Dense(
            units=out_channels,
            use_bias=use_bias,
            input_dim=in_channels,
            name="fc")
        if self.use_bn:
            self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation, name="activ")

    def call(self, x, training=None):
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


class ConvBlock1d(nn.Layer):
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
    strides : int or
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    force_same : bool, default False
        Whether to forcibly set `same` padding in convolution.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 force_same=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(ConvBlock1d, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            force_same=force_same,
            data_format=data_format,
            name="conv")
        if self.use_bn:
            self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation, name="activ")

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1(in_channels,
            out_channels,
            strides=1,
            groups=1,
            use_bias=False,
            data_format="channels_last",
            **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        **kwargs)


def conv3x3(in_channels,
            out_channels,
            strides=1,
            padding=1,
            dilation=1,
            groups=1,
            use_bias=False,
            data_format="channels_last",
            **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        **kwargs)


def depthwise_conv3x3(channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      use_bias=False,
                      data_format="channels_last",
                      **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=channels,
        use_bias=use_bias,
        data_format=data_format,
        **kwargs)


class ConvBlock(nn.Layer):
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
    force_same : bool, default False
        Whether to forcibly set `same` padding in convolution.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 force_same=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            force_same=force_same,
            data_format=data_format,
            name="conv")
        if self.use_bn:
            self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation, name="activ")

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
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
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def conv3x3_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def conv5x5_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=2,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def conv7x7_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation="relu",
                  data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def dwconv_block(in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def dwconv3x3_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation="relu",
                    data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def dwconv5x5_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=2,
                    dilation=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation="relu",
                    data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


class DwsConvBlock(nn.Layer):
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
    dw_force_same : bool, default False
        Whether to forcibly set `same` padding in depthwise convolution block.
    pw_force_same : bool, default False
        Whether to forcibly set `same` padding in pointwise convolution block.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default 'relu'
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default 'relu'
        Activation function after the pointwise convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 dw_force_same=False,
                 pw_force_same=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_eps=1e-5,
                 dw_activation="relu",
                 pw_activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        self.dw_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            use_bias=use_bias,
            force_same=dw_force_same,
            use_bn=dw_use_bn,
            bn_eps=bn_eps,
            activation=dw_activation,
            data_format=data_format,
            name="dw_conv")
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            force_same=pw_force_same,
            use_bn=pw_use_bn,
            bn_eps=bn_eps,
            activation=pw_activation,
            data_format=data_format,
            name="pw_conv")

    def call(self, x, training=None):
        x = self.dw_conv(x, training=training)
        x = self.pw_conv(x, training=training)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=1,
                     dilation=1,
                     use_bias=False,
                     bn_eps=1e-5,
                     dw_activation="relu",
                     pw_activation="relu",
                     data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default 'relu'
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default 'relu'
        Activation function after the pointwise convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        data_format=data_format,
        **kwargs)


class PreConvBlock(nn.Layer):
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
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 return_preact=False,
                 activate=True,
                 data_format="channels_last",
                 **kwargs):
        super(PreConvBlock, self).__init__(**kwargs)
        self.return_preact = return_preact
        self.activate = activate
        self.use_bn = use_bn

        if self.use_bn:
            self.bn = BatchNorm(
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = nn.ReLU()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        if self.use_bn:
            x = self.bn(x, training=training)
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
                      return_preact=False,
                      activate=True,
                      data_format="channels_last",
                      **kwargs):
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
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        use_bias=use_bias,
        use_bn=use_bn,
        return_preact=return_preact,
        activate=activate,
        data_format=data_format,
        **kwargs)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      use_bias=False,
                      use_bn=True,
                      return_preact=False,
                      activate=True,
                      data_format="channels_last",
                      **kwargs):
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
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        return_preact=return_preact,
        activate=activate,
        data_format=data_format,
        **kwargs)


class Deconv2d(nn.Layer):
    """
    Standard deconvolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    out_padding : int or tuple/list of 2 int, default 0
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 out_padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 data_format="channels_last",
                 **kwargs):
        super(Deconv2d, self).__init__(**kwargs)
        assert (dilation == 1)
        assert (groups == 1)
        assert (in_channels is not None)

        if isinstance(padding, int):
            padding = (padding, padding)

        self.use_crop = (padding[0] > 0) or (padding[1] > 0)
        if self.use_crop:
            self.crop = nn.Cropping2D(
                cropping=padding,
                data_format=data_format,
                name="crop")

        self.conv = nn.Conv2DTranspose(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            output_padding=out_padding,
            data_format=data_format,
            dilation_rate=dilation,
            use_bias=use_bias,
            name="conv")

    def call(self, x):
        x = self.conv(x)
        if self.use_crop:
            x = self.crop(x)
        return x


class DeconvBlock(nn.Layer):
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
    out_padding : int or tuple/list of 2 int, default 0
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = Deconv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            out_padding=out_padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            data_format=data_format,
            name="conv")
        if self.use_bn:
            self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation, name="activ")

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


def channel_shuffle(x,
                    groups,
                    data_format):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.
    data_format : str
        The ordering of the dimensions in tensors.

    Returns:
    -------
    Tensor
        Resulted tensor.
    """
    x_shape = x.get_shape().as_list()
    if is_channels_first(data_format):
        channels = x_shape[1]
        height = x_shape[2]
        width = x_shape[3]
    else:
        height = x_shape[1]
        width = x_shape[2]
        channels = x_shape[3]

    assert (channels % groups == 0)
    channels_per_group = channels // groups

    if is_channels_first(data_format):
        x = tf.reshape(x, shape=(-1, groups, channels_per_group, height, width))
        x = tf.transpose(x, perm=(0, 2, 1, 3, 4))
        x = tf.reshape(x, shape=(-1, channels, height, width))
    else:
        x = tf.reshape(x, shape=(-1, height, width, groups, channels_per_group))
        x = tf.transpose(x, perm=(0, 1, 2, 4, 3))
        x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class ChannelShuffle(nn.Layer):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 groups,
                 data_format="channels_last",
                 **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.groups = groups
        self.data_format = data_format

    def call(self, x):
        return channel_shuffle(x, groups=self.groups, data_format=self.data_format)


def channel_shuffle2(x,
                     channels_per_group,
                     data_format):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.
    The alternative version.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    channels_per_group : int
        Number of groups.
    data_format : str
        Number of channels per group.

    Returns:
    -------
    keras.Tensor
        Resulted tensor.
    """
    x_shape = x.get_shape().as_list()
    if is_channels_first(data_format):
        channels = x_shape[1]
        height = x_shape[2]
        width = x_shape[3]
    else:
        height = x_shape[1]
        width = x_shape[2]
        channels = x_shape[3]

    assert (channels % channels_per_group == 0)
    groups = channels // channels_per_group

    if is_channels_first(data_format):
        x = tf.reshape(x, shape=(-1, channels_per_group, groups, height, width))
        x = tf.transpose(x, perm=(0, 2, 1, 3, 4))
        x = tf.reshape(x, shape=(-1, channels, height, width))
    else:
        x = tf.reshape(x, shape=(-1, height, width, channels_per_group, groups))
        x = tf.transpose(x, perm=(0, 1, 2, 4, 3))
        x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class ChannelShuffle2(nn.Layer):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 groups,
                 data_format="channels_last",
                 **kwargs):
        super(ChannelShuffle2, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.channels_per_group = channels // groups
        self.data_format = data_format

    def call(self, x):
        return channel_shuffle2(x, channels_per_group=self.channels_per_group, data_format=self.data_format)


class SEBlock(nn.Layer):
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
    activation : function, or str, or nn.Layer, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Layer, default 'sigmoid'
        Activation function after the last convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 mid_channels=None,
                 round_mid=False,
                 use_conv=True,
                 mid_activation="relu",
                 out_activation="sigmoid",
                 data_format="channels_last",
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.use_conv = use_conv
        self.data_format = data_format
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        if use_conv:
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                use_bias=True,
                data_format=data_format,
                name="conv1")
        else:
            self.fc1 = nn.Dense(
                units=mid_channels,
                input_dim=channels,
                name="fc1")
        self.activ = get_activation_layer(mid_activation, name="activ")
        if use_conv:
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels,
                use_bias=True,
                data_format=data_format,
                name="conv2")
        else:
            self.fc2 = nn.Dense(
                units=channels,
                input_dim=mid_channels,
                name="fc2")
        self.sigmoid = get_activation_layer(out_activation, name="sigmoid")

    def call(self, x, training=None):
        w = self.pool(x)
        if self.use_conv:
            axis = -1 if is_channels_first(self.data_format) else 1
            w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            axis = -1 if is_channels_first(self.data_format) else 1
            w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        x = x * w
        return x


class SABlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 out_channels,
                 groups,
                 radix,
                 reduction=4,
                 min_channels=32,
                 use_conv=True,
                 bn_eps=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(SABlock, self).__init__(**kwargs)
        self.groups = groups
        self.radix = radix
        self.use_conv = use_conv
        self.data_format = data_format
        self.axis = get_channel_axis(data_format)
        in_channels = out_channels * radix
        mid_channels = max(in_channels // reduction, min_channels)

        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        if use_conv:
            self.conv1 = conv1x1(
                in_channels=out_channels,
                out_channels=mid_channels,
                use_bias=True,
                data_format=data_format,
                name="conv1")
        else:
            self.fc1 = nn.Dense(
                units=mid_channels,
                input_dim=out_channels,
                name="fc1")
        self.bn = BatchNorm(
            epsilon=bn_eps,
            data_format=data_format,
            name="bn")
        self.activ = nn.ReLU()
        if use_conv:
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=in_channels,
                use_bias=True,
                data_format=data_format,
                name="conv2")
        else:
            self.fc2 = nn.Dense(
                units=in_channels,
                input_dim=mid_channels,
                name="fc2")
        self.softmax = nn.Softmax(axis=1)

    def call(self, x, training=None):
        x_shape = x.get_shape().as_list()
        # batch = x_shape[0]
        if is_channels_first(self.data_format):
            channels = x_shape[1]
            height = x_shape[2]
            width = x_shape[3]
            x = tf.reshape(x, shape=(-1, self.radix, channels // self.radix, height, width))
            w = tf.math.reduce_sum(x, axis=1)
        else:
            height = x_shape[1]
            width = x_shape[2]
            channels = x_shape[3]
            x = tf.reshape(x, shape=(-1, height, width, self.radix, channels // self.radix))
            w = tf.math.reduce_sum(x, axis=-2)

        w = self.pool(w)
        if self.use_conv:
            axis = -1 if is_channels_first(self.data_format) else 1
            w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.bn(w, training=training)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = tf.reshape(w, shape=(-1, self.groups, self.radix, channels // self.groups // self.radix))
        w = tf.transpose(w, perm=(0, 2, 1, 3))
        w = self.softmax(w)
        if is_channels_first(self.data_format):
            w = tf.reshape(w, shape=(-1, self.radix, channels // self.radix, 1, 1))
        else:
            w = tf.reshape(w, shape=(-1, 1, 1, self.radix, channels // self.radix))
        x = x * w
        if is_channels_first(self.data_format):
            x = tf.math.reduce_sum(x, axis=1)
        else:
            x = tf.math.reduce_sum(x, axis=-2)
        return x


class SAConvBlock(nn.Layer):
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
    force_same : bool, default False
        Whether to forcibly set `same` padding in convolution.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    radix : int, default 2
        Number of splits within a cardinal group.
    reduction : int, default 4
        Squeeze reduction value.
    min_channels : int, default 32
        Minimal number of squeezed channels.
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 force_same=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 radix=2,
                 reduction=4,
                 min_channels=32,
                 use_conv=True,
                 data_format="channels_last",
                 **kwargs):
        super(SAConvBlock, self).__init__(**kwargs)
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=(out_channels * radix),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=(groups * radix),
            use_bias=use_bias,
            force_same=force_same,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=activation,
            data_format=data_format,
            name="conv")
        self.att = SABlock(
            out_channels=out_channels,
            groups=groups,
            radix=radix,
            reduction=reduction,
            min_channels=min_channels,
            use_conv=use_conv,
            bn_eps=bn_eps,
            data_format=data_format,
            name="att")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.att(x, training=training)
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


class PixelShuffle(nn.Layer):
    """
    Pixel-shuffle operation from 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
    Convolutional Neural Network,' https://arxiv.org/abs/1609.05158.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 scale_factor,
                 data_format="channels_last",
                 **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.data_format = data_format

    def call(self, x, training=None):
        if not tf.executing_eagerly():
            return PixelShuffle2(scale_factor=self.scale_factor, data_format=self.data_format)(x)

        f1 = self.scale_factor
        f2 = self.scale_factor

        x_shape = x.get_shape().as_list()
        if is_channels_first(self.data_format):
            channels = x_shape[1]
            height = x_shape[2]
            width = x_shape[3]
        else:
            height = x_shape[1]
            width = x_shape[2]
            channels = x_shape[3]

        assert (channels % f1 % f2 == 0)
        new_channels = channels // f1 // f2

        if is_channels_first(self.data_format):
            x = tf.reshape(x, shape=(-1, new_channels, f1 * f2, height, width))
            x = tf.reshape(x, shape=(-1, new_channels, f1, f2, height, width))
            x = tf.transpose(x, perm=(0, 1, 4, 2, 5, 3))
            x = tf.reshape(x, shape=(-1, new_channels, height * f1, width * f2))
        else:
            x = tf.reshape(x, shape=(-1, height, width, new_channels, f1 * f2))
            x = tf.reshape(x, shape=(-1, height, width, new_channels, f1, f2))
            x = tf.transpose(x, perm=(0, 1, 4, 2, 5, 3))
            x = tf.reshape(x, shape=(-1, height * f1, width * f2, new_channels))

        return x


class PixelShuffle2(nn.Layer):
    """
    Pixel-shuffle operation from 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
    Convolutional Neural Network,' https://arxiv.org/abs/1609.05158. Alternative implementation.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 scale_factor,
                 data_format="channels_last",
                 **kwargs):
        super(PixelShuffle2, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.data_format = data_format

    def call(self, x, training=None):
        tf_data_format = "NCHW" if is_channels_first(self.data_format) else "NHWC"
        x = tf.nn.depth_to_space(input=x, block_size=self.scale_factor, data_format=tf_data_format)
        return x


class DucBlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor,
                 data_format="channels_last",
                 **kwargs):
        super(DucBlock, self).__init__(**kwargs)
        mid_channels = (scale_factor * scale_factor) * out_channels

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv")
        self.pix_shuffle = PixelShuffle(
            scale_factor=scale_factor,
            data_format=data_format,
            name="pix_shuffle")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pix_shuffle(x)
        return x


class Identity(nn.Layer):
    """
    Identity layer.
    """
    def __init__(self,
                 **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, x, training=None):
        return x


class SimpleSequential(nn.Layer):
    """
    A sequential layer that can be used instead of tf.keras.Sequential.
    """
    def __init__(self,
                 **kwargs):
        super(SimpleSequential, self).__init__(**kwargs)
        self.children = []

    def __getitem__(self, i):
        return self.children[i]

    def __len__(self):
        return len(self.children)

    def add(self, layer):
        layer._name = "{}/{}".format(self.name, layer._name)
        self.children.append(layer)

    def call(self, x, training=None):
        for block in self.children:
            x = block(x, training=training)
        return x


class ParametricSequential(SimpleSequential):
    """
    A sequential container for layers with parameters.
    Layers will be executed in the order they are added.
    """
    def __init__(self,
                 **kwargs):
        super(ParametricSequential, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        for block in self.children:
            x = block(x, **kwargs)
        return x


class DualPathSequential(SimpleSequential):
    """
    A sequential container for layers with dual inputs/outputs.
    Layers will be executed in the order they are added.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first layers with single input/output.
    last_ordinals : int, default 0
        Number of the final layers with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a layer.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal layer.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda block, x1, x2, training: block(x1, x2, training)),
                 dual_path_scheme_ordinal=(lambda block, x1, x2, training: (block(x1, training), x2)),
                 **kwargs):
        super(DualPathSequential, self).__init__(**kwargs)
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def call(self, x1, x2=None, training=None):
        length = len(self.children)
        for i, block in enumerate(self.children):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(block, x1, x2, training)
            else:
                x1, x2 = self.dual_path_scheme(block, x1, x2, training)
        if self.return_two:
            return x1, x2
        else:
            return x1


class Concurrent(SimpleSequential):
    """
    A container for concatenation of layers.

    Parameters:
    ----------
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    merge_type : str, default None
        Type of branch merging.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 stack=False,
                 merge_type=None,
                 data_format="channels_last",
                 **kwargs):
        super(Concurrent, self).__init__(**kwargs)
        assert (merge_type is None) or (merge_type in ["cat", "stack", "sum"])
        self.axis = get_channel_axis(data_format)
        if merge_type is not None:
            self.merge_type = merge_type
        else:
            self.merge_type = "stack" if stack else "cat"

    def call(self, x, training=None):
        out = []
        for block in self.children:
            out.append(block(x, training=training))
        if self.merge_type == "stack":
            out = tf.stack(out, axis=self.axis)
        elif self.merge_type == "cat":
            out = tf.concat(out, axis=self.axis)
        elif self.merge_type == "sum":
            out = tf.stack(out, axis=self.axis)
            out = tf.math.reduce_sum(out, axis=self.axis)
        else:
            raise NotImplementedError()
        return out


class SequentialConcurrent(SimpleSequential):
    """
    A sequential container with concatenated outputs.
    Blocks will be executed in the order they are added.

    Parameters:
    ----------
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    cat_input : bool, default True
        Whether to concatenate input tensor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 stack=False,
                 cat_input=True,
                 data_format="channels_last",
                 **kwargs):
        super(SequentialConcurrent, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        self.stack = stack
        self.cat_input = cat_input

    def call(self, x, training=None):
        out = [x] if self.cat_input else []
        for block in self.children:
            x = block(x, training=training)
            out.append(x)
        if self.stack:
            out = tf.stack(out, axis=self.axis)
        else:
            out = tf.concat(out, axis=self.axis)
        return out


class ParametricConcurrent(SimpleSequential):
    """
    A container for concatenation of layers with parameters.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(ParametricConcurrent, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)

    def call(self, x, **kwargs):
        out = []
        for block in self.children:
            out.append(block(x, **kwargs))
        out = tf.concat(out, axis=self.axis)
        return out


class MultiOutputSequential(SimpleSequential):
    """
    A sequential container with multiple outputs.
    Layers will be executed in the order they are added.

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

    def call(self, x, **kwargs):
        outs = []
        for block in self.children:
            x = block(x, **kwargs)
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
    A sequential container with multiple inputs and single/multiple outputs.
    Modules will be executed in the order they are added.

    Parameters:
    ----------
    merge_type : str, default None
        Type of branch merging.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 merge_type=None,
                 data_format="channels_last",
                 **kwargs):
        super(ParallelConcurent, self).__init__(**kwargs)
        assert (merge_type is None) or (merge_type in ["list", "cat", "stack", "sum"])
        self.axis = get_channel_axis(data_format)
        self.merge_type = merge_type

    def call(self, x, training=None):
        out = []
        for block, xi in zip(self.children, x):
            out.append(block(xi, training=training))
        if self.merge_type == "list":
            pass
        elif self.merge_type == "stack":
            out = tf.stack(out, axis=self.axis)
        elif self.merge_type == "cat":
            out = tf.concat(out, axis=self.axis)
        elif self.merge_type == "sum":
            out = tf.stack(out, axis=self.axis)
            out = tf.math.reduce_sum(out, axis=self.axis)
        else:
            raise NotImplementedError()
        return out


class DualPathParallelConcurent(SimpleSequential):
    """
    A sequential container with multiple dual-path inputs and single/multiple outputs.
    Modules will be executed in the order they are added.

    Parameters:
    ----------
    merge_type : str, default 'list'
        Type of branch merging.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 merge_type="list",
                 data_format="channels_last",
                 **kwargs):
        super(DualPathParallelConcurent, self).__init__(**kwargs)
        assert (merge_type is None) or (merge_type in ["list", "cat", "stack", "sum"])
        self.axis = get_channel_axis(data_format)
        self.merge_type = merge_type

    def call(self, x1, x2, training=None):
        x1_out = []
        x2_out = []
        for block, x1i, x2i in zip(self.children, x1, x2):
            y1i, y2i = block(x1i, x2i, training=training)
            x1_out.append(y1i)
            x2_out.append(y2i)
        if self.merge_type == "list":
            pass
        elif self.merge_type == "stack":
            x1_out = tf.stack(x1_out, axis=self.axis)
            x2_out = tf.stack(x2_out, axis=self.axis)
        elif self.merge_type == "cat":
            x1_out = tf.concat(x1_out, axis=self.axis)
            x2_out = tf.concat(x2_out, axis=self.axis)
        elif self.merge_type == "sum":
            x1_out = tf.math.reduce_sum(tf.stack(x1_out, axis=self.axis), axis=self.axis)
            x2_out = tf.math.reduce_sum(tf.stack(x2_out, axis=self.axis), axis=self.axis)
        else:
            raise NotImplementedError()
        return x1_out, x2_out


class NormActivation(nn.Layer):
    """
    Activation block with preliminary batch normalization. It's used by itself as the final block in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(NormActivation, self).__init__(**kwargs)
        assert (in_channels is not None)

        self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        self.activ = get_activation_layer(activation, name="activ")

    def call(self, x, training=None):
        x = self.bn(x, training=training)
        x = self.activ(x)
        return x


class InterpolationBlock(nn.Layer):
    """
    Bilinear interpolation block.

    Parameters:
    ----------
    scale_factor : int, default 1
        Multiplier for spatial size.
    out_size : tuple of 2 int, default None
        Spatial size of the output tensor for the interpolation operation.
    up : bool, default True
        Whether to upsample or downsample.
    interpolation : str, default 'bilinear'
        Interpolation mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 scale_factor=1,
                 out_size=None,
                 up=True,
                 interpolation="bilinear",
                 data_format="channels_last",
                 **kwargs):
        super(InterpolationBlock, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.out_size = out_size
        self.up = up
        self.data_format = data_format
        self.method = tf.image.ResizeMethod.BILINEAR if interpolation == "bilinear" else\
            tf.image.ResizeMethod.NEAREST_NEIGHBOR

    def call(self, x, size=None, training=None):
        out_size = self.calc_out_size(x) if size is None else size
        if is_channels_first(self.data_format):
            x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = tf.image.resize(
            images=x,
            size=out_size,
            method=self.method)
        if is_channels_first(self.data_format):
            x = tf.transpose(x, perm=[0, 3, 1, 2])
        return x

    def calc_out_size(self, x):
        if self.out_size is not None:
            return self.out_size
        in_size = get_im_size(x, data_format=self.data_format)
        if self.up:
            return tuple(s * self.scale_factor for s in in_size)
        else:
            return tuple(s // self.scale_factor for s in in_size)


class Hourglass(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 down_seq,
                 up_seq,
                 skip_seq,
                 merge_type="add",
                 return_first_skip=False,
                 data_format="channels_last",
                 **kwargs):
        super(Hourglass, self).__init__(**kwargs)
        self.depth = len(down_seq)
        assert (merge_type in ["cat", "add"])
        assert (len(up_seq) == self.depth)
        assert (len(skip_seq) in (self.depth, self.depth + 1))
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.extra_skip = (len(skip_seq) == self.depth + 1)
        self.axis = get_channel_axis(data_format)

        self.down_seq = down_seq
        self.up_seq = up_seq
        self.skip_seq = skip_seq

    def _merge(self, x, y):
        if y is not None:
            if self.merge_type == "cat":
                x = tf.concat([x, y], axis=self.axis)
            elif self.merge_type == "add":
                x = x + y
        return x

    def call(self, x, training=None):
        y = None
        down_outs = [x]
        for down_module in self.down_seq.children:
            x = down_module(x, training=training)
            down_outs.append(x)
        for i in range(len(down_outs)):
            if i != 0:
                y = down_outs[self.depth - i]
                skip_module = self.skip_seq[self.depth - i]
                y = skip_module(y, training=training)
                x = self._merge(x, y)
            if i != len(down_outs) - 1:
                if (i == 0) and self.extra_skip:
                    skip_module = self.skip_seq[self.depth]
                    x = skip_module(x, training=training)
                up_module = self.up_seq[self.depth - 1 - i]
                x = up_module(x, training=training)
        if self.return_first_skip:
            return x, y
        else:
            return x


class HeatmapMaxDetBlock(nn.Layer):
    """
    Heatmap maximum detector block (for human pose estimation task).

    Parameters:
    ----------
    tune : bool, default True
        Whether to tune point positions.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 tune=True,
                 data_format="channels_last",
                 **kwargs):
        super(HeatmapMaxDetBlock, self).__init__(**kwargs)
        self.tune = tune
        self.data_format = data_format

    def call(self, x, training=None):
        # if not tf.executing_eagerly():
        #     channels = x.shape[1] if is_channels_first(self.data_format) else x.shape[3]
        #     return x[:, :channels, :3, 0]
        vector_dim = 2
        x_shape = x.get_shape().as_list()
        batch = x_shape[0]
        if is_channels_first(self.data_format):
            channels = x_shape[1]
            in_size = x_shape[2:]
            heatmap_vector = tf.reshape(x, shape=(batch, channels, -1))
        else:
            channels = x_shape[3]
            in_size = x_shape[1:3]
            heatmap_vector = tf.reshape(x, shape=(batch, -1, channels))
            heatmap_vector = tf.transpose(heatmap_vector, perm=(0, 2, 1))
        indices = tf.cast(tf.expand_dims(tf.cast(tf.math.argmax(heatmap_vector, axis=vector_dim), np.int32),
                                         axis=vector_dim), np.float32)
        scores = tf.math.reduce_max(heatmap_vector, axis=vector_dim, keepdims=True)
        scores_mask = tf.cast(tf.math.greater(scores, 0.0), np.float32)
        pts_x = (indices % in_size[1]) * scores_mask
        pts_y = (indices // in_size[1]) * scores_mask
        pts = tf.concat([pts_x, pts_y, scores], axis=vector_dim)
        if self.tune:
            pts = pts.numpy()
            for b in range(batch):
                for k in range(channels):
                    hm = x[b, k, :, :] if is_channels_first(self.data_format) else x[b, :, :, k]
                    px = int(pts[b, k, 0])
                    py = int(pts[b, k, 1])
                    if (0 < px < in_size[1] - 1) and (0 < py < in_size[0] - 1):
                        pts[b, k, 0] += np.sign(hm[py, px + 1] - hm[py, px - 1]) * 0.25
                        pts[b, k, 1] += np.sign(hm[py + 1, px] - hm[py - 1, px]) * 0.25
            pts = tf.convert_to_tensor(pts)
        return pts
