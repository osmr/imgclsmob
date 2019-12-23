"""
    SE-ResNeXt for ImageNet-1K, implemented in Keras.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['seresnext', 'seresnext50_32x4d', 'seresnext101_32x4d', 'seresnext101_64x4d']

import os
from keras import layers as nn
from keras.models import Model
from .common import conv1x1_block, se_block, is_channels_first, flatten
from .resnet import res_init_block
from .resnext import resnext_bottleneck


def seresnext_unit(x,
                   in_channels,
                   out_channels,
                   strides,
                   cardinality,
                   bottleneck_width,
                   name="seresnext_unit"):
    """
    SE-ResNeXt unit.

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
    name : str, default 'seresnext_unit'
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

    x = se_block(
        x=x,
        channels=out_channels,
        name=name + "/se")

    x = nn.add([x, identity], name=name + "/add")

    activ = nn.Activation("relu", name=name + "/activ")
    x = activ(x)
    return x


def seresnext(channels,
              init_block_channels,
              cardinality,
              bottleneck_width,
              in_channels=3,
              in_size=(224, 224),
              classes=1000):
    """
    SE-ResNeXt model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
            x = seresnext_unit(
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


def get_seresnext(blocks,
                  cardinality,
                  bottleneck_width,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".keras", "models"),
                  **kwargs):
    """
    Create SE-ResNeXt model with specific parameters.

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

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported SE-ResNeXt with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = seresnext(
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


def seresnext50_32x4d(**kwargs):
    """
    SE-ResNeXt-50 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="seresnext50_32x4d", **kwargs)


def seresnext101_32x4d(**kwargs):
    """
    SE-ResNeXt-101 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="seresnext101_32x4d", **kwargs)


def seresnext101_64x4d(**kwargs):
    """
    SE-ResNeXt-101 (64x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="seresnext101_64x4d", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        seresnext50_32x4d,
        seresnext101_32x4d,
        seresnext101_64x4d,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != seresnext50_32x4d or weight_count == 27559896)
        assert (model != seresnext101_32x4d or weight_count == 48955416)
        assert (model != seresnext101_64x4d or weight_count == 88232984)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
