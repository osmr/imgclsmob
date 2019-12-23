"""
    ResNeXt for ImageNet-1K, implemented in Keras.
    Original paper: 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
"""

__all__ = ['resnext', 'resnext14_16x4d', 'resnext14_32x2d', 'resnext14_32x4d', 'resnext26_16x4d', 'resnext26_32x2d',
           'resnext26_32x4d', 'resnext38_32x4d', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
           'resnext_bottleneck']

import os
import math
from keras import layers as nn
from keras.models import Model
from .common import conv1x1_block, conv3x3_block, is_channels_first, flatten
from .resnet import res_init_block


def resnext_bottleneck(x,
                       in_channels,
                       out_channels,
                       strides,
                       cardinality,
                       bottleneck_width,
                       bottleneck_factor=4,
                       name="resnext_bottleneck"):
    """
    ResNeXt bottleneck block for residual path in ResNeXt unit.

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
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    name : str, default 'resnext_bottleneck'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_channels = out_channels // bottleneck_factor
    D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
    group_width = cardinality * D

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=group_width,
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=group_width,
        out_channels=group_width,
        strides=strides,
        groups=cardinality,
        name=name + "/conv2")
    x = conv1x1_block(
        x=x,
        in_channels=group_width,
        out_channels=out_channels,
        activation=None,
        name=name + "/conv3")
    return x


def resnext_unit(x,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 name="resnext_unit"):
    """
    ResNeXt unit with residual connection.

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
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    name : str, default 'resnext_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        identity = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            activation=None,
            name=name + "/identity_conv")
    else:
        identity = x

    x = resnext_bottleneck(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        name=name + "/body")

    x = nn.add([x, identity], name=name + "/add")

    activ = nn.Activation("relu", name=name + "/activ")
    x = activ(x)
    return x


def resnext(channels,
            init_block_channels,
            cardinality,
            bottleneck_width,
            in_channels=3,
            in_size=(224, 224),
            classes=1000):
    """
    ResNeXt model from 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
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

    x = res_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            x = resnext_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                cardinality=cardinality,
                bottleneck_width=bottleneck_width,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
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


def get_resnext(blocks,
                cardinality,
                bottleneck_width,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".keras", "models"),
                **kwargs):
    """
    Create ResNeXt model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    if blocks == 14:
        layers = [1, 1, 1, 1]
    elif blocks == 26:
        layers = [2, 2, 2, 2]
    elif blocks == 38:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported ResNeXt with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = resnext(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
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


def resnext14_16x4d(**kwargs):
    """
    ResNeXt-14 (16x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=14, cardinality=16, bottleneck_width=4, model_name="resnext14_16x4d", **kwargs)


def resnext14_32x2d(**kwargs):
    """
    ResNeXt-14 (32x2d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=14, cardinality=32, bottleneck_width=2, model_name="resnext14_32x2d", **kwargs)


def resnext14_32x4d(**kwargs):
    """
    ResNeXt-14 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=14, cardinality=32, bottleneck_width=4, model_name="resnext14_32x4d", **kwargs)


def resnext26_16x4d(**kwargs):
    """
    ResNeXt-26 (16x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=26, cardinality=16, bottleneck_width=4, model_name="resnext26_16x4d", **kwargs)


def resnext26_32x2d(**kwargs):
    """
    ResNeXt-26 (32x2d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=26, cardinality=32, bottleneck_width=2, model_name="resnext26_32x2d", **kwargs)


def resnext26_32x4d(**kwargs):
    """
    ResNeXt-26 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=26, cardinality=32, bottleneck_width=4, model_name="resnext26_32x4d", **kwargs)


def resnext38_32x4d(**kwargs):
    """
    ResNeXt-38 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=38, cardinality=32, bottleneck_width=4, model_name="resnext38_32x4d", **kwargs)


def resnext50_32x4d(**kwargs):
    """
    ResNeXt-50 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="resnext50_32x4d", **kwargs)


def resnext101_32x4d(**kwargs):
    """
    ResNeXt-101 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="resnext101_32x4d", **kwargs)


def resnext101_64x4d(**kwargs):
    """
    ResNeXt-101 (64x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="resnext101_64x4d", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        resnext14_16x4d,
        resnext14_32x2d,
        resnext14_32x4d,
        resnext26_16x4d,
        resnext26_32x2d,
        resnext26_32x4d,
        resnext38_32x4d,
        resnext50_32x4d,
        resnext101_32x4d,
        resnext101_64x4d,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnext14_16x4d or weight_count == 7127336)
        assert (model != resnext14_32x2d or weight_count == 7029416)
        assert (model != resnext14_32x4d or weight_count == 9411880)
        assert (model != resnext26_16x4d or weight_count == 10119976)
        assert (model != resnext26_32x2d or weight_count == 9924136)
        assert (model != resnext26_32x4d or weight_count == 15389480)
        assert (model != resnext38_32x4d or weight_count == 21367080)
        assert (model != resnext50_32x4d or weight_count == 25028904)
        assert (model != resnext101_32x4d or weight_count == 44177704)
        assert (model != resnext101_64x4d or weight_count == 83455272)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
