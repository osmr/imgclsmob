"""
    MobileNetV2, implemented in Keras.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
"""

__all__ = ['mobilenetv2', 'mobilenetv2_w1', 'mobilenetv2_w3d4', 'mobilenetv2_wd2', 'mobilenetv2_wd4']

import os
from keras import layers as nn
from keras.models import Model
from .common import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, is_channels_first, flatten


def linear_bottleneck(x,
                      in_channels,
                      out_channels,
                      strides,
                      expansion,
                      name="linear_bottleneck"):
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
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    expansion : bool
        Whether do expansion of channels.
    name : str, default 'linear_bottleneck'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    residual = (in_channels == out_channels) and (strides == 1)
    mid_channels = in_channels * 6 if expansion else in_channels

    if residual:
        identity = x

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        activation="relu6",
        name=name + "/conv1")
    x = dwconv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        strides=strides,
        activation="relu6",
        name=name + "/conv2")
    x = conv1x1_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activation=None,
        activate=False,
        name=name + "/conv3")

    if residual:
        x = nn.add([x, identity], name=name + "/add")

    return x


def mobilenetv2(channels,
                init_block_channels,
                final_block_channels,
                in_channels=3,
                in_size=(224, 224),
                classes=1000):
    """
    MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if is_channels_first() else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = conv3x3_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        strides=2,
        activation="relu6",
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            expansion = (i != 0) or (j != 0)
            x = linear_bottleneck(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                expansion=expansion,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=final_block_channels,
        activation="relu6",
        name="features/final_block")
    in_channels = final_block_channels
    x = nn.AvgPool2D(
        pool_size=7,
        strides=1,
        name="features/final_pool")(x)

    x = conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=classes,
        use_bias=False,
        name="output")
    # x = nn.Flatten()(x)
    x = flatten(x)

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_mobilenetv2(width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join('~', '.keras', 'models'),
                    **kwargs):
    """
    Create MobileNetV2 model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [[]])

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = int(final_block_channels * width_scale)

    net = mobilenetv2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def mobilenetv2_w1(**kwargs):
    """
    1.0 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=1.0, model_name="mobilenetv2_w1", **kwargs)


def mobilenetv2_w3d4(**kwargs):
    """
    0.75 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=0.75, model_name="mobilenetv2_w3d4", **kwargs)


def mobilenetv2_wd2(**kwargs):
    """
    0.5 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=0.5, model_name="mobilenetv2_wd2", **kwargs)


def mobilenetv2_wd4(**kwargs):
    """
    0.25 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=0.25, model_name="mobilenetv2_wd4", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        mobilenetv2_w1,
        mobilenetv2_w3d4,
        mobilenetv2_wd2,
        mobilenetv2_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenetv2_w1 or weight_count == 3504960)
        assert (model != mobilenetv2_w3d4 or weight_count == 2627592)
        assert (model != mobilenetv2_wd2 or weight_count == 1964736)
        assert (model != mobilenetv2_wd4 or weight_count == 1516392)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
