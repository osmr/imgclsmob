"""
    DenseNet for ImageNet-1K, implemented in Keras.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

__all__ = ['densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

import os
from keras import layers as nn
from keras.models import Model
from .common import pre_conv1x1_block, pre_conv3x3_block, is_channels_first, get_channel_axis, flatten
from .preresnet import preres_init_block, preres_activation


def dense_unit(x,
               in_channels,
               out_channels,
               dropout_rate,
               name="dense_unit"):
    """
    DenseNet unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : bool
        Parameter of Dropout layer. Faction of the input units to drop.
    name : str, default 'dense_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor.
    """
    bn_size = 4
    inc_channels = out_channels - in_channels
    mid_channels = inc_channels * bn_size

    identity = x

    x = pre_conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        name=name + "/conv1")
    x = pre_conv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=inc_channels,
        name=name + "/conv2")

    use_dropout = (dropout_rate != 0.0)
    if use_dropout:
        x = nn.Dropout(
            rate=dropout_rate,
            name=name + "dropout")(x)

    x = nn.concatenate([identity, x], axis=get_channel_axis(), name=name + "/concat")
    return x


def transition_block(x,
                     in_channels,
                     out_channels,
                     name="transition_block"):
    """
    DenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'transition_block'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor.
    """
    x = pre_conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        name=name + "/conv")
    x = nn.AvgPool2D(
        pool_size=2,
        strides=2,
        padding="valid",
        name=name + "/pool")(x)
    return x


def densenet(channels,
             init_block_channels,
             dropout_rate=0.0,
             in_channels=3,
             in_size=(224, 224),
             classes=1000):
    """
    DenseNet model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if is_channels_first() else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = preres_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        if i != 0:
            x = transition_block(
                x=x,
                in_channels=in_channels,
                out_channels=(in_channels // 2),
                name="features/stage{}/trans{}".format(i + 1, i + 1))
            in_channels = in_channels // 2
        for j, out_channels in enumerate(channels_per_stage):
            x = dense_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = preres_activation(
        x=x,
        name="features/post_activ")
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


def get_densenet(blocks,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join('~', '.keras', 'models'),
                 **kwargs):
    """
    Create DenseNet model with specific parameters.

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

    if blocks == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif blocks == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif blocks == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif blocks == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported DenseNet version with number of layers {}".format(blocks))

    from functools import reduce
    channels = reduce(lambda xi, yi:
                      xi + [reduce(lambda xj, yj:
                                   xj + [xj[-1] + yj],
                                   [growth_rate] * yi,
                                   [xi[-1][-1] // 2])[1:]],
                      layers,
                      [[init_block_channels * 2]])[1:]

    net = densenet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def densenet121(**kwargs):
    """
    DenseNet-121 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_densenet(blocks=121, model_name="densenet121", **kwargs)


def densenet161(**kwargs):
    """
    DenseNet-161 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_densenet(blocks=161, model_name="densenet161", **kwargs)


def densenet169(**kwargs):
    """
    DenseNet-169 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_densenet(blocks=169, model_name="densenet169", **kwargs)


def densenet201(**kwargs):
    """
    DenseNet-201 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_densenet(blocks=201, model_name="densenet201", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        densenet121,
        densenet161,
        densenet169,
        densenet201,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != densenet121 or weight_count == 7978856)
        assert (model != densenet161 or weight_count == 28681000)
        assert (model != densenet169 or weight_count == 14149480)
        assert (model != densenet201 or weight_count == 20013928)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
