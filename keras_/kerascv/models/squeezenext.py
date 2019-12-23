"""
    SqueezeNext for ImageNet-1K, implemented in Keras.
    Original paper: 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
"""

__all__ = ['squeezenext', 'sqnxt23_w1', 'sqnxt23_w3d2', 'sqnxt23_w2', 'sqnxt23v5_w1', 'sqnxt23v5_w3d2', 'sqnxt23v5_w2']

import os
from keras import layers as nn
from keras.models import Model
from .common import maxpool2d, conv_block, conv1x1_block, conv7x7_block, is_channels_first, flatten


def sqnxt_unit(x,
               in_channels,
               out_channels,
               strides,
               name="sqnxt_unit"):
    """
    SqueezeNext unit.

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
    name : str, default 'sqnxt_unit'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if strides == 2:
        reduction_den = 1
        resize_identity = True
    elif in_channels > out_channels:
        reduction_den = 4
        resize_identity = True
    else:
        reduction_den = 2
        resize_identity = False

    if resize_identity:
        identity = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            use_bias=True,
            name=name + "/identity_conv")
    else:
        identity = x

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=(in_channels // reduction_den),
        strides=strides,
        use_bias=True,
        name=name + "/conv1")
    x = conv1x1_block(
        x=x,
        in_channels=(in_channels // reduction_den),
        out_channels=(in_channels // (2 * reduction_den)),
        use_bias=True,
        name=name + "/conv2")
    x = conv_block(
        x=x,
        in_channels=(in_channels // (2 * reduction_den)),
        out_channels=(in_channels // reduction_den),
        kernel_size=(1, 3),
        strides=1,
        padding=(0, 1),
        use_bias=True,
        name=name + "/conv3")
    x = conv_block(
        x=x,
        in_channels=(in_channels // reduction_den),
        out_channels=(in_channels // reduction_den),
        kernel_size=(3, 1),
        strides=1,
        padding=(1, 0),
        use_bias=True,
        name=name + "/conv4")
    x = conv1x1_block(
        x=x,
        in_channels=(in_channels // reduction_den),
        out_channels=out_channels,
        use_bias=True,
        name=name + "/conv5")

    x = nn.add([x, identity], name=name + "/add")
    x = nn.Activation("relu", name=name + "/final_activ")(x)
    return x


def sqnxt_init_block(x,
                     in_channels,
                     out_channels,
                     name="sqnxt_init_block"):
    """
    ResNet specific initial block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'sqnxt_init_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = conv7x7_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        padding=1,
        use_bias=True,
        name=name + "/conv")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        ceil_mode=True,
        name=name + "/pool")
    return x


def squeezenext(channels,
                init_block_channels,
                final_block_channels,
                in_channels=3,
                in_size=(224, 224),
                classes=1000):
    """
    SqueezeNext model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

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
    input_shape = (in_channels, in_size[0], in_size[1]) if is_channels_first() else\
        (in_size[0], in_size[1], in_channels)
    input = nn.Input(shape=input_shape)

    x = sqnxt_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            x = sqnxt_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=final_block_channels,
        use_bias=True,
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


def get_squeezenext(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".keras", "models"),
                    **kwargs):
    """
    Create SqueezeNext model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('23' or '23v5').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 64
    final_block_channels = 128
    channels_per_layers = [32, 64, 128, 256]

    if version == '23':
        layers = [6, 6, 8, 1]
    elif version == '23v5':
        layers = [2, 4, 14, 1]
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        final_block_channels = int(final_block_channels * width_scale)

    net = squeezenext(
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


def sqnxt23_w1(**kwargs):
    """
    1.0-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23", width_scale=1.0, model_name="sqnxt23_w1", **kwargs)


def sqnxt23_w3d2(**kwargs):
    """
    1.5-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23", width_scale=1.5, model_name="sqnxt23_w3d2", **kwargs)


def sqnxt23_w2(**kwargs):
    """
    2.0-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23", width_scale=2.0, model_name="sqnxt23_w2", **kwargs)


def sqnxt23v5_w1(**kwargs):
    """
    1.0-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23v5", width_scale=1.0, model_name="sqnxt23v5_w1", **kwargs)


def sqnxt23v5_w3d2(**kwargs):
    """
    1.5-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23v5", width_scale=1.5, model_name="sqnxt23v5_w3d2", **kwargs)


def sqnxt23v5_w2(**kwargs):
    """
    2.0-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23v5", width_scale=2.0, model_name="sqnxt23v5_w2", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        sqnxt23_w1,
        sqnxt23_w3d2,
        sqnxt23_w2,
        sqnxt23v5_w1,
        sqnxt23v5_w3d2,
        sqnxt23v5_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sqnxt23_w1 or weight_count == 724056)
        assert (model != sqnxt23_w3d2 or weight_count == 1511824)
        assert (model != sqnxt23_w2 or weight_count == 2583752)
        assert (model != sqnxt23v5_w1 or weight_count == 921816)
        assert (model != sqnxt23v5_w3d2 or weight_count == 1953616)
        assert (model != sqnxt23v5_w2 or weight_count == 3366344)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
