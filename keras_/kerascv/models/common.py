"""
    Common routines for models in Keras.
"""

__all__ = ['is_channels_first', 'get_channel_axis', 'update_keras_shape', 'flatten', 'batchnorm', 'maxpool2d',
           'avgpool2d', 'conv2d', 'conv1x1', 'conv3x3', 'depthwise_conv3x3', 'conv_block', 'conv1x1_block',
           'conv3x3_block', 'conv7x7_block', 'dwconv3x3_block', 'pre_conv_block', 'pre_conv1x1_block',
           'pre_conv3x3_block', 'channel_shuffle_lambda', 'se_block']

import math
import numpy as np
from inspect import isfunction
from keras.layers import BatchNormalization
from keras import backend as K
from keras import layers as nn


def is_channels_first():
    """
    Is tested data format channels first.

    Returns
    -------
    bool
        A flag.
    """
    return K.image_data_format() == "channels_first"


def get_channel_axis():
    """
    Get channel axis.

    Returns
    -------
    int
        Channel axis.
    """
    return 1 if is_channels_first() else -1


def update_keras_shape(x):
    """
    Update Keras shape property.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    """
    if not hasattr(x, "_keras_shape"):
        x._keras_shape = tuple([int(d) if d != 0 else None for d in x.shape])


def flatten(x,
            reshape=False):
    """
    Flattens the input to two dimensional.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    reshape : bool, default False
        Whether do reshape instead of flatten.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if not is_channels_first():
        def channels_last_flatten(z):
            z = K.permute_dimensions(z, pattern=(0, 3, 1, 2))
            z = K.reshape(z, shape=(-1, np.prod(K.int_shape(z)[1:])))
            update_keras_shape(z)
            return z
        return nn.Lambda(channels_last_flatten)(x)
    else:
        if reshape:
            x = nn.Reshape((-1,))(x)
        else:
            x = nn.Flatten()(x)
        return x


def batchnorm(x,
              momentum=0.9,
              epsilon=1e-5,
              name=None):
    """
    Batch normalization layer.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    name : str, default None
        Layer name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if K.backend() == "mxnet":
        x = GluonBatchNormalization(
            momentum=momentum,
            epsilon=epsilon,
            name=name)(x)
    else:
        x = nn.BatchNormalization(
            axis=get_channel_axis(),
            momentum=momentum,
            epsilon=epsilon,
            name=name)(x)
    return x


def maxpool2d(x,
              pool_size,
              strides,
              padding=0,
              ceil_mode=False,
              name=None):
    """
    Max pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the pooling.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    name : str, default None
        Layer name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    assert (padding[0] == 0) or (padding[0] == (pool_size[0] - 1) // 2)
    assert (padding[1] == 0) or (padding[1] == (pool_size[1] - 1) // 2)

    padding_ke = "valid" if padding[0] == 0 else "same"

    if K.backend() == "tensorflow":
        if ceil_mode:
            height = int(x.shape[2])
            out_height = float(height + 2 * padding[0] - pool_size[0]) / strides[0] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            width = int(x.shape[3])
            out_width = float(width + 2 * padding[1] - pool_size[1]) / strides[1] + 1.0
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)

        if (padding[0] > 0) or (padding[1] > 0):
            import tensorflow as tf
            x = nn.Lambda(
                (lambda z: tf.pad(z, [[0, 0], [0, 0], list(padding), list(padding)], mode="REFLECT"))
                if is_channels_first() else
                (lambda z: tf.pad(z, [[0, 0], list(padding), list(padding), [0, 0]], mode="REFLECT")))(x)
        padding_ke = "valid"
    else:
        if ceil_mode:
            padding0 = 0 if padding_ke == "valid" else strides[0] // 2
            height = x._keras_shape[2 if is_channels_first() else 1]
            out_height = float(height + 2 * padding0 - pool_size[0]) / strides[0] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                assert (strides[0] <= 3)
                padding_ke = "same"

    x = nn.MaxPool2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding_ke,
        name=name + "/pool")(x)
    return x


def avgpool2d(x,
              pool_size,
              strides,
              padding=0,
              ceil_mode=False,
              name=None):
    """
    Average pooling operation for two dimensional (spatial) data.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    pool_size : int or tuple/list of 2 int
        Size of the max pooling windows.
    strides : int or tuple/list of 2 int
        Strides of the pooling.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    name : str, default None
        Layer name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    assert (padding[0] == 0) or (padding[0] == (pool_size[0] - 1) // 2)
    assert (padding[1] == 0) or (padding[1] == (pool_size[1] - 1) // 2)

    padding_ke = "valid" if padding[0] == 0 else "same"

    if K.backend() == "tensorflow":
        if ceil_mode:
            height = int(x.shape[2])
            out_height = float(height + 2 * padding[0] - pool_size[0]) / strides[0] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            width = int(x.shape[3])
            out_width = float(width + 2 * padding[1] - pool_size[1]) / strides[1] + 1.0
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)

        if (padding[0] > 0) or (padding[1] > 0):
            import tensorflow as tf
            x = nn.Lambda(
                (lambda z: tf.pad(z, [[0, 0], [0, 0], list(padding), list(padding)], mode="REFLECT"))
                if is_channels_first() else
                (lambda z: tf.pad(z, [[0, 0], list(padding), list(padding), [0, 0]], mode="REFLECT")))(x)

        x = nn.AvgPool2D(
            pool_size=pool_size,
            strides=1,
            padding="valid",
            name=name + "/pool")(x)

        if (strides[0] > 1) or (strides[1] > 1):
            x = nn.AvgPool2D(
                pool_size=1,
                strides=strides,
                padding="valid",
                name=name + "/stride")(x)
        return x

    x = nn.AvgPool2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding_ke,
        name=name + "/pool")(x)
    return x


def conv2d(x,
           in_channels,
           out_channels,
           kernel_size,
           strides=1,
           padding=0,
           dilation=1,
           groups=1,
           use_bias=True,
           name="conv2d"):
    """
    Convolution 2D layer wrapper.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    name : str, default 'conv2d'
        Layer name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    extra_pad = False
    if K.backend() == "tensorflow":
        if (padding[0] > 0) or (padding[1] > 0):
            import tensorflow as tf
            x = nn.Lambda(
                (lambda z: tf.pad(z, [[0, 0], [0, 0], list(padding), list(padding)]))
                if is_channels_first() else
                (lambda z: tf.pad(z, [[0, 0], list(padding), list(padding), [0, 0]])))(x)
            if not ((padding[0] == padding[1]) and (kernel_size[0] == kernel_size[1]) and
                    (kernel_size[0] // 2 == padding[0])):
                extra_pad = True
        padding_ke = "valid"
    else:
        if (padding[0] == padding[1]) and (padding[0] == 0):
            padding_ke = "valid"
        elif (padding[0] == padding[1]) and (kernel_size[0] == kernel_size[1]) and (kernel_size[0] // 2 == padding[0]):
            padding_ke = "same"
        else:
            x = nn.ZeroPadding2D(
                padding=padding,
                name=name + "/pad")(x)
            padding_ke = "valid"
            extra_pad = True

    if groups == 1:
        if extra_pad:
            name = name + "/conv"
        x = nn.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding_ke,
            dilation_rate=dilation,
            use_bias=use_bias,
            name=name)(x)
    elif (groups == out_channels) and (out_channels == in_channels):
        assert (dilation[0] == 1) and (dilation[1] == 1)
        if extra_pad:
            name = name + "/conv"
        x = nn.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding_ke,
            use_bias=use_bias,
            name=name)(x)
    else:
        assert (in_channels % groups == 0)
        assert (out_channels % groups == 0)
        none_batch = (x._keras_shape[0] is None)
        in_group_channels = in_channels // groups
        out_group_channels = out_channels // groups
        group_list = []
        for gi in range(groups):
            xi = nn.Lambda(
                (lambda z: z[:, gi * in_group_channels:(gi + 1) * in_group_channels, :, :])
                if is_channels_first() else
                (lambda z: z[:, :, :, gi * in_group_channels:(gi + 1) * in_group_channels]))(x)
            xi = nn.Conv2D(
                filters=out_group_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding_ke,
                dilation_rate=dilation,
                use_bias=use_bias,
                name=name + "/convgroup{}".format(gi + 1))(xi)
            group_list.append(xi)
        x = nn.concatenate(group_list, axis=get_channel_axis(), name=name + "/concat")
        if none_batch and (x._keras_shape[0] is not None):
            x._keras_shape = (None, ) + x._keras_shape[1:]

    return x


def conv1x1(x,
            in_channels,
            out_channels,
            strides=1,
            groups=1,
            use_bias=False,
            name="conv1x1"):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    name : str, default 'conv1x1'
        Layer name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        name=name)


def conv3x3(x,
            in_channels,
            out_channels,
            strides=1,
            padding=1,
            groups=1,
            name="conv3x3"):
    """
    Convolution 3x3 layer.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    name : str, default 'conv3x3'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=False,
        name=name)


def depthwise_conv3x3(x,
                      channels,
                      strides,
                      name="depthwise_conv3x3"):
    """
    Depthwise convolution 3x3 layer.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    name : str, default 'depthwise_conv3x3'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return conv2d(
        x=x,
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=channels,
        use_bias=False,
        name=name)


def conv_block(x,
               in_channels,
               out_channels,
               kernel_size,
               strides,
               padding,
               dilation=1,
               groups=1,
               use_bias=False,
               activation="relu",
               activate=True,
               name="conv_block"):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    name : str, default 'conv_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        name=name + "/conv")
    x = batchnorm(
        x=x,
        name=name + "/bn")
    if activate:
        assert (activation is not None)
        if isfunction(activation):
            x = activation()(x)
        elif isinstance(activation, str):
            if activation == "relu":
                x = nn.Activation("relu", name=name + "/activ")(x)
            elif activation == "relu6":
                x = nn.ReLU(max_value=6.0, name=name + "/activ")(x)
            else:
                raise NotImplementedError()
        else:
            x = activation(x)
    return x


def conv1x1_block(x,
                  in_channels,
                  out_channels,
                  strides=1,
                  groups=1,
                  use_bias=False,
                  activation="relu",
                  activate=True,
                  name="conv1x1_block"):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    name : str, default 'conv1x1_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        activation=activation,
        activate=activate,
        name=name)


def conv3x3_block(x,
                  in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  activation="relu",
                  activate=True,
                  name="conv3x3_block"):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    name : str, default 'conv3x3_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        activation=activation,
        activate=activate,
        name=name)


def conv7x7_block(x,
                  in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  activation="relu",
                  activate=True,
                  name="conv7x7_block"):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    name : str, default 'conv7x7_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        activation=activation,
        activate=activate,
        name=name)


def dwconv3x3_block(x,
                    in_channels,
                    out_channels,
                    strides,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    activation="relu",
                    activate=True,
                    name="dwconv3x3_block"):
    """
    3x3 depthwise version of the standard convolution block with ReLU6 activation.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    name : str, default 'dwconv3x3_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return conv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        activation=activation,
        activate=activate,
        name=name)


def pre_conv_block(x,
                   in_channels,
                   out_channels,
                   kernel_size,
                   strides,
                   padding,
                   return_preact=False,
                   name="pre_conv_block"):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    name : str, default 'pre_conv_block'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    x = batchnorm(
        x=x,
        name=name + "/bn")
    x = nn.Activation("relu", name=name + "/activ")(x)
    if return_preact:
        x_pre_activ = x
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name + "/conv")
    if return_preact:
        return x, x_pre_activ
    else:
        return x


def pre_conv1x1_block(x,
                      in_channels,
                      out_channels,
                      strides=1,
                      return_preact=False,
                      name="preres_conv1x1"):
    """
    1x1 version of the pre-activated convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    return_preact : bool, default False
        Whether return pre-activation.
    name : str, default 'preres_conv1x1'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    return pre_conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        return_preact=return_preact,
        name=name)


def pre_conv3x3_block(x,
                      in_channels,
                      out_channels,
                      strides=1,
                      return_preact=False,
                      name="pre_conv3x3_block"):
    """
    3x3 version of the pre-activated convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    return_preact : bool, default False
        Whether return pre-activation.
    name : str, default 'pre_conv3x3_block'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    return pre_conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        return_preact=return_preact,
        name=name)


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    groups : int
        Number of groups.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """

    if is_channels_first():
        batch, channels, height, width = x._keras_shape
    else:
        batch, height, width, channels = x._keras_shape

    # assert (channels % groups == 0)
    channels_per_group = channels // groups

    if is_channels_first():
        x = K.reshape(x, shape=(-1, groups, channels_per_group, height, width))
        x = K.permute_dimensions(x, pattern=(0, 2, 1, 3, 4))
        x = K.reshape(x, shape=(-1, channels, height, width))
    else:
        x = K.reshape(x, shape=(-1, height, width, groups, channels_per_group))
        x = K.permute_dimensions(x, pattern=(0, 1, 2, 4, 3))
        x = K.reshape(x, shape=(-1, height, width, channels))

    update_keras_shape(x)
    return x


def channel_shuffle_lambda(channels,
                           groups,
                           **kwargs):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.

    Returns
    -------
    Layer
        Channel shuffle layer.
    """
    assert (channels % groups == 0)

    return nn.Lambda(channel_shuffle, arguments={"groups": groups}, **kwargs)


def se_block(x,
             channels,
             reduction=16,
             name="se_block"):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    name : str, default 'se_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    assert(len(x._keras_shape) == 4)
    mid_cannels = channels // reduction
    pool_size = x._keras_shape[2:4] if is_channels_first() else x._keras_shape[1:3]

    w = nn.AvgPool2D(
        pool_size=pool_size,
        name=name + "/pool")(x)
    w = conv1x1(
        x=w,
        in_channels=channels,
        out_channels=mid_cannels,
        use_bias=True,
        name=name + "/conv1")
    w = nn.Activation("relu", name=name + "/relu")(w)
    w = conv1x1(
        x=w,
        in_channels=mid_cannels,
        out_channels=channels,
        use_bias=True,
        name=name + "/conv2")
    w = nn.Activation("sigmoid", name=name + "/sigmoid")(w)
    x = nn.multiply([x, w], name=name + "/mul")
    return x


class GluonBatchNormalization(BatchNormalization):
    """
    Batch normalization layer wrapper for implementation of the Gluon type of BatchNorm default parameters.

    Parameters
    ----------
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center : bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale : bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer : str, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer : str, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer : str, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer : str, default 'ones'
        Initializer for the moving variance.
    beta_regularizer : str or None, default None
        Optional regularizer for the beta weight.
    gamma_regularizer : str or None, default None
        Optional regularizer for the gamma weight.
    beta_constraint : str or None, default None
        Optional constraint for the beta weight.
    gamma_constraint : str or None, default None
        Optional constraint for the gamma weight.
    fix_gamma : bool, default False
        Fix gamma while training.
    """
    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 fix_gamma=False,
                 **kwargs):
        super(GluonBatchNormalization, self).__init__(
            axis=get_channel_axis(),
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs)
        self.fix_gamma = fix_gamma

    def call(self, inputs, training=None):
        if K.backend() == "mxnet":

            from keras.backend.mxnet_backend import keras_mxnet_symbol, KerasSymbol
            import mxnet as mx

            @keras_mxnet_symbol
            def gluon_batchnorm(x,
                                gamma,
                                beta,
                                moving_mean,
                                moving_var,
                                momentum=0.9,
                                axis=1,
                                epsilon=1e-5,
                                fix_gamma=False):
                """
                Apply native MXNet/Gluon batch normalization on x with given moving_mean, moving_var, beta and gamma.


                Parameters
                ----------
                x : keras.backend tensor/variable/symbol
                    Input tensor/variable/symbol.
                gamma : keras.backend tensor/variable/symbol
                    Tensor by which to scale the input.
                beta : keras.backend tensor/variable/symbol
                    Tensor by which to center the input.
                moving_mean : keras.backend tensor/variable/symbol
                    Moving mean.
                moving_var : keras.backend tensor/variable/symbol
                    Moving variance.
                momentum : float, default 0.9
                    Momentum for the moving average.
                axis : int, default 1
                    Axis along which BatchNorm is applied. Axis usually represent axis of 'channels'. MXNet follows
                    'channels_first'.
                epsilon : float, default 1e-5
                    Small float added to variance to avoid dividing by zero.
                fix_gamma : bool, default False
                    Fix gamma while training.

                Returns
                -------
                keras.backend tensor/variable/symbol
                    Resulted tensor/variable/symbol.
                """
                if isinstance(x, KerasSymbol):
                    x = x.symbol
                if isinstance(moving_mean, KerasSymbol):
                    moving_mean = moving_mean.symbol
                if isinstance(moving_var, KerasSymbol):
                    moving_var = moving_var.symbol
                if isinstance(beta, KerasSymbol):
                    beta = beta.symbol
                if isinstance(gamma, KerasSymbol):
                    gamma = gamma.symbol
                return KerasSymbol(mx.sym.BatchNorm(
                    data=x,
                    gamma=gamma,
                    beta=beta,
                    moving_mean=moving_mean,
                    moving_var=moving_var,
                    momentum=momentum,
                    axis=axis,
                    eps=epsilon,
                    fix_gamma=fix_gamma))

            return gluon_batchnorm(
                x=inputs,
                gamma=self.gamma,
                beta=self.beta,
                moving_mean=self.moving_mean,
                moving_var=self.moving_variance,
                momentum=self.momentum,
                axis=self.axis,
                epsilon=self.epsilon,
                fix_gamma=self.fix_gamma)
        else:
            super(GluonBatchNormalization, self).call(inputs, training)
