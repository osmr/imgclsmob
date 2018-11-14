"""
    MnasNet, implemented in Keras.
    Original paper: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,' https://arxiv.org/abs/1807.11626.
"""

__all__ = ['mnasnet_model', 'mnasnet']

import os
from keras import backend as K
from keras import layers as nn
from keras.models import Model
from .common import conv2d, GluonBatchNormalization


def conv_block(x,
               in_channels,
               out_channels,
               kernel_size,
               strides,
               padding,
               groups,
               activate,
               name="conv_block"):
    """
    Standard convolution block with Batch normalization and ReLU activation.

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
    groups : int
        Number of groups.
    activate : bool
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
        groups=groups,
        use_bias=False,
        name=name + "/conv")
    x = GluonBatchNormalization(name=name + "/bn")(x)
    if activate:
        x = nn.Activation("relu", name=name + "/activ")(x)
    return x


def conv1x1_block(x,
                  in_channels,
                  out_channels,
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
        strides=1,
        padding=0,
        groups=1,
        activate=activate,
        name=name)


def dwconv_block(x,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 activate=True,
                 name="dwconv_block"):
    """
    Depthwise version of the standard convolution block.

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
    activate : bool, default True
        Whether activate the convolution block.
    name : str, default 'dwconv_block'
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
        kernel_size=kernel_size,
        strides=strides,
        padding=(kernel_size // 2),
        groups=out_channels,
        activate=activate,
        name=name)


def dws_conv_block(x,
                   in_channels,
                   out_channels,
                   name="dws_conv_block"):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'dws_conv_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = dwconv_block(
        x=x,
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=3,
        strides=1,
        name=name + "/dw_conv")
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        name=name + "/pw_conv")
    return x


def mnas_unit(x,
              in_channels,
              out_channels,
              kernel_size,
              strides,
              expansion_factor,
              name="mnas_unit"):
    """
    So-called 'Linear Bottleneck' layer. It is used as a MobileNetV2 unit.

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
    expansion_factor : int
        Factor for expansion of channels.
    name : str, default 'mnas_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    residual = (in_channels == out_channels) and (strides == 1)
    mid_channels = in_channels * expansion_factor

    if residual:
        identity = x

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        activate=True,
        name=name + "/conv1")
    x = dwconv_block(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        kernel_size=kernel_size,
        strides=strides,
        activate=True,
        name=name + "/conv2")
    x = conv1x1_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activate=False,
        name=name + "/conv3")

    if residual:
        x = nn.add([x, identity], name=name + "/add")

    return x


def mnas_init_block(x,
                    in_channels,
                    out_channels_list,
                    name="mnas_init_block"):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels_list : list of 2 int
        Numbers of output channels.
    name : str, default 'mnas_init_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels_list[0],
        kernel_size=3,
        strides=2,
        padding=1,
        groups=1,
        activate=True,
        name=name + "/conv1")
    x = dws_conv_block(
        x=x,
        in_channels=out_channels_list[0],
        out_channels=out_channels_list[1],
        name=name + "/conv2")
    return x


def mnasnet_model(channels,
                  init_block_channels,
                  final_block_channels,
                  kernel_sizes,
                  expansion_factors,
                  in_channels=3,
                  in_size=(224, 224),
                  classes=1000):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Numbers of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list of list of int
        Number of kernel sizes for each unit.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if K.image_data_format() == 'channels_first' else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = mnas_init_block(
        x=input,
        in_channels=in_channels,
        out_channels_list=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels[-1]
    for i, channels_per_stage in enumerate(channels):
        kernel_sizes_per_stage = kernel_sizes[i]
        expansion_factors_per_stage = expansion_factors[i]
        for j, out_channels in enumerate(channels_per_stage):
            kernel_size = kernel_sizes_per_stage[j]
            expansion_factor = expansion_factors_per_stage[j]
            strides = 2 if (j == 0) else 1
            x = mnas_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                expansion_factor=expansion_factor,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=final_block_channels,
        activate=True,
        name="features/final_block")
    in_channels = final_block_channels
    x = nn.AvgPool2D(
        pool_size=7,
        strides=1,
        name="features/final_pool")(x)

    x = nn.Flatten()(x)
    x = nn.Dense(
        units=classes,
        input_dim=in_channels,
        name="output")(x)

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_mnasnet(model_name=None,
                pretrained=False,
                root=os.path.join('~', '.keras', 'models'),
                **kwargs):
    """
    Create MnasNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    init_block_channels = [32, 16]
    final_block_channels = 1280
    layers = [3, 3, 3, 2, 4, 1]
    downsample = [1, 1, 1, 0, 1, 0]
    channels_per_layers = [24, 40, 80, 96, 192, 320]
    expansion_factors_per_layers = [3, 3, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 5, 5, 3, 5, 3]
    default_kernel_size = 3

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] + [default_kernel_size] * (y[1] - 1)] if y[2] != 0 else x[:-1] + [
        x[-1] + [y[0]] + [default_kernel_size] * (y[1] - 1)], zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])

    net = mnasnet_model(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        expansion_factors=expansion_factors,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def mnasnet(**kwargs):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(model_name="mnasnet", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        mnasnet,
    ]

    for model in models:
        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mnasnet or weight_count == 4308816)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
