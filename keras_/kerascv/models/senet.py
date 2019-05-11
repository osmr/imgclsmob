"""
    SENet, implemented in Keras.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['senet', 'senet52', 'senet103', 'senet154']

import os
import math
from keras import layers as nn
from keras.models import Model
from .common import conv1x1_block, conv3x3_block, se_block, is_channels_first, flatten


def senet_bottleneck(x,
                     in_channels,
                     out_channels,
                     strides,
                     cardinality,
                     bottleneck_width,
                     name="senet_bottleneck"):
    """
    SENet bottleneck block for residual path in SENet unit.

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
    name : str, default 'senet_bottleneck'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_channels = out_channels // 4
    D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
    group_width = cardinality * D
    group_width2 = group_width // 2

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=group_width2,
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=group_width2,
        out_channels=group_width,
        strides=strides,
        groups=cardinality,
        name=name + "/conv2")
    x = conv1x1_block(
        x=x,
        in_channels=group_width,
        out_channels=out_channels,
        activate=False,
        name=name + "/conv3")
    return x


def senet_unit(x,
               in_channels,
               out_channels,
               strides,
               cardinality,
               bottleneck_width,
               identity_conv3x3,
               name="senet_unit"):
    """
    SENet unit.

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
    identity_conv3x3 : bool, default False
        Whether to use 3x3 convolution in the identity link.
    name : str, default 'senet_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        if identity_conv3x3:
            identity = conv3x3_block(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activate=False,
                name=name + "/identity_conv")
        else:
            identity = conv1x1_block(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activate=False,
                name=name + "/identity_conv")
    else:
        identity = x

    x = senet_bottleneck(
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

    activ = nn.Activation('relu', name=name + "/activ")
    x = activ(x)
    return x


def senet_init_block(x,
                     in_channels,
                     out_channels,
                     name="senet_init_block"):
    """
    SENet specific initial block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'senet_init_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_channels = out_channels // 2

    x = conv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        strides=2,
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        name=name + "/conv2")
    x = conv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        name=name + "/conv3")
    x = nn.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same',
        name=name + "/pool")(x)
    return x


def senet(channels,
          init_block_channels,
          cardinality,
          bottleneck_width,
          in_channels=3,
          in_size=(224, 224),
          classes=1000):
    """
    SENet model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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

    x = senet_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        identity_conv3x3 = (i != 0)
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            x = senet_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                cardinality=cardinality,
                bottleneck_width=bottleneck_width,
                identity_conv3x3=identity_conv3x3,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = nn.AvgPool2D(
        pool_size=7,
        strides=1,
        name="features/final_pool")(x)

    # x = nn.Flatten()(x)
    x = flatten(x)
    x = nn.Dropout(
        rate=0.2,
        name="output/dropout")(x)
    x = nn.Dense(
        units=classes,
        input_dim=in_channels,
        name="output/fc")(x)

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_senet(blocks,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".keras", "models"),
              **kwargs):
    """
    Create SENet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    if blocks == 52:
        layers = [3, 4, 6, 3]
        cardinality = 32
    elif blocks == 103:
        layers = [3, 4, 23, 3]
        cardinality = 32
    elif blocks == 154:
        layers = [3, 8, 36, 3]
        cardinality = 64
    else:
        raise ValueError("Unsupported SENet with number of blocks: {}".format(blocks))

    bottleneck_width = 4
    init_block_channels = 128
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = senet(
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


def senet52(**kwargs):
    """
    SENet-52 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_senet(blocks=52, model_name="senet52", **kwargs)


def senet103(**kwargs):
    """
    SENet-103 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_senet(blocks=103, model_name="senet103", **kwargs)


def senet154(**kwargs):
    """
    SENet-154 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_senet(blocks=154, model_name="senet154", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        senet52,
        senet103,
        senet154,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != senet52 or weight_count == 44659416)
        assert (model != senet103 or weight_count == 60963096)
        assert (model != senet154 or weight_count == 115088984)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
