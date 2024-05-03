"""
    IGCV3 for ImageNet-1K, implemented in Keras.
    Original paper: 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.
"""

__all__ = ['igcv3', 'igcv3_w1', 'igcv3_w3d4', 'igcv3_wd2', 'igcv3_wd4']

import os
from keras import layers as nn
from keras.models import Model
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, channel_shuffle_lambda, is_channels_first, flatten


def inv_res_unit(x,
                 in_channels,
                 out_channels,
                 strides,
                 expansion,
                 name="inv_res_unit"):
    """
    So-called 'Inverted Residual Unit' layer.

    Parameters
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
    name : str, default 'inv_res_unit'
        Unit name.

    Returns:
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    residual = (in_channels == out_channels) and (strides == 1)
    mid_channels = in_channels * 6 if expansion else in_channels
    groups = 2

    if residual:
        identity = x

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        groups=groups,
        activation=None,
        name=name + "/conv1")
    x = channel_shuffle_lambda(
        channels=mid_channels,
        groups=groups,
        name=name + "/c_shuffle")(x)
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
        groups=groups,
        activation=None,
        name=name + "/conv3")

    if residual:
        x = nn.add([x, identity], name=name + "/add")

    return x


def igcv3(channels,
          init_block_channels,
          final_block_channels,
          in_channels=3,
          in_size=(224, 224),
          classes=1000):
    """
    IGCV3 model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters
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
    input_shape = (in_channels, in_size[0], in_size[1]) if is_channels_first() else\
        (in_size[0], in_size[1], in_channels)
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
            x = inv_res_unit(
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

    # x = nn.Flatten()(x)
    x = flatten(x)
    x = nn.Dense(
        units=classes,
        input_dim=in_channels,
        name="output")(x)

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_igcv3(width_scale,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".keras", "models"),
              **kwargs):
    """
    Create IGCV3-D model with specific parameters.

    Parameters
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
    layers = [1, 4, 6, 8, 6, 6, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [[]])

    if width_scale != 1.0:
        def make_even(x):
            return x if (x % 2 == 0) else x + 1
        channels = [[make_even(int(cij * width_scale)) for cij in ci] for ci in channels]
        init_block_channels = make_even(int(init_block_channels * width_scale))
        if width_scale > 1.0:
            final_block_channels = make_even(int(final_block_channels * width_scale))

    net = igcv3(
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


def igcv3_w1(**kwargs):
    """
    IGCV3-D 1.0x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=1.0, model_name="igcv3_w1", **kwargs)


def igcv3_w3d4(**kwargs):
    """
    IGCV3-D 0.75x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.75, model_name="igcv3_w3d4", **kwargs)


def igcv3_wd2(**kwargs):
    """
    IGCV3-D 0.5x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.5, model_name="igcv3_wd2", **kwargs)


def igcv3_wd4(**kwargs):
    """
    IGCV3-D 0.25x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.25, model_name="igcv3_wd4", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        igcv3_w1,
        igcv3_w3d4,
        igcv3_wd2,
        igcv3_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != igcv3_w1 or weight_count == 3491688)
        assert (model != igcv3_w3d4 or weight_count == 2638084)
        assert (model != igcv3_wd2 or weight_count == 1985528)
        assert (model != igcv3_wd4 or weight_count == 1534020)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
