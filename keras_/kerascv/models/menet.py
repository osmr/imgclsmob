"""
    MENet, implemented in Keras.
    Original paper: 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications,'
    https://arxiv.org/abs/1803.09127.
"""

__all__ = ['menet', 'menet108_8x1_g3', 'menet128_8x1_g4', 'menet160_8x1_g8', 'menet228_12x1_g3', 'menet256_12x1_g4',
           'menet348_12x1_g3', 'menet352_12x1_g8', 'menet456_24x1_g3']

import os
from keras import backend as K
from keras import layers as nn
from keras.models import Model
from .common import conv2d, conv1x1, channel_shuffle_lambda, GluonBatchNormalization


def depthwise_conv3x3(x,
                      channels,
                      strides,
                      name="depthwise_conv3x3"):
    """
    Depthwise convolution 3x3 layer. This is exactly the same layer as in ShuffleNet.

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


def group_conv1x1(x,
                  in_channels,
                  out_channels,
                  groups,
                  name="group_conv1x1"):
    """
    Group convolution 1x1 layer. This is exactly the same layer as in ShuffleNet.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    name : str, default 'depthwise_conv3x3'
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
        kernel_size=1,
        groups=groups,
        use_bias=False,
        name=name)


def conv3x3(x,
            in_channels,
            out_channels,
            strides,
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
    strides : int or tuple/list of 2 int
        Strides of the convolution.
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
        padding=1,
        use_bias=False,
        name=name)


def me_unit(x,
            in_channels,
            out_channels,
            side_channels,
            groups,
            downsample,
            ignore_group,
            name="me_unit"):
    """
    MENet unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    side_channels : int
        Number of side channels.
    groups : int
        Number of groups in convolution layers.
    downsample : bool
        Whether do downsample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    name : str, default 'me_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_channels = out_channels // 4

    if downsample:
        out_channels -= in_channels

    identity = x

    # pointwise group convolution 1
    x = group_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        groups=(1 if ignore_group else groups),
        name=name + "/compress_conv1")
    x = GluonBatchNormalization(name=name + "/compress_bn1")(x)
    x = nn.Activation("relu", name=name + "/compress_activ")(x)

    x = channel_shuffle_lambda(
        channels=mid_channels,
        groups=groups,
        name=name + "/c_shuffle")(x)

    # merging
    y = conv1x1(
        out_channels=side_channels,
        name=name + "/s_merge_conv")(x)
    y = GluonBatchNormalization(name=name + "/s_merge_bn")(y)
    y = nn.Activation("relu", name=name + "/s_merge_activ")(y)

    # depthwise convolution (bottleneck)
    x = depthwise_conv3x3(
        x=x,
        channels=mid_channels,
        strides=(2 if downsample else 1),
        name=name + "/dw_conv2")
    x = GluonBatchNormalization(name=name + "/dw_bn2")(x)

    # evolution
    y = conv3x3(
        x=y,
        in_channels=side_channels,
        out_channels=side_channels,
        strides=(2 if downsample else 1),
        name=name + "/s_conv")
    y = GluonBatchNormalization(name=name + "/s_conv_bn")(y)
    y = nn.Activation("relu", name=name + "/s_conv_activ")(y)

    y = conv1x1(
        out_channels=mid_channels,
        name=name + "/s_evolve_conv")(y)
    y = GluonBatchNormalization(name=name + "/s_evolve_bn")(y)
    y = nn.Activation('sigmoid', name=name + "/s_evolve_activ")(y)

    x = nn.multiply([x, y], name=name + "/mul")

    # pointwise group convolution 2
    x = group_conv1x1(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        groups=groups,
        name=name + "/expand_conv3")
    x = GluonBatchNormalization(name=name + "/expand_bn3")(x)

    x._keras_shape = tuple([d if d != 0 else None for d in x.shape])

    if downsample:
        identity = nn.AvgPool2D(
            pool_size=3,
            strides=2,
            padding="same",
            name=name + "/avgpool")(identity)

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = nn.concatenate([x, identity], axis=channel_axis, name=name + "/concat")
    else:
        x = nn.add([x, identity], name=name + "/add")

    x = nn.Activation("relu", name=name + "/final_activ")(x)
    return x


def me_init_block(x,
                  in_channels,
                  out_channels,
                  name="me_init_block"):
    """
    MENet specific initial block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'shuffle_init_block'
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
        kernel_size=3,
        strides=2,
        padding=1,
        use_bias=False,
        name=name + "/conv")
    x = GluonBatchNormalization(name=name + "/bn")(x)
    x = nn.Activation("relu", name=name + "/activ")(x)
    x = nn.MaxPool2D(
        pool_size=3,
        strides=2,
        padding="same",
        name=name + "/pool")(x)
    return x


def menet(channels,
          init_block_channels,
          side_channels,
          groups,
          in_channels=3,
          classes=1000):
    """
    ShuffleNet model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    side_channels : int
        Number of side channels in a ME-unit.
    groups : int
        Number of groups in convolution layers.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if K.image_data_format() == 'channels_first' else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = me_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            downsample = (j == 0)
            ignore_group = (i == 0) and (j == 0)
            x = me_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                side_channels=side_channels,
                groups=groups,
                downsample=downsample,
                ignore_group=ignore_group,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
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
    return model


def get_menet(first_stage_channels,
              side_channels,
              groups,
              model_name=None,
              pretrained=False,
              root=os.path.join('~', '.keras', 'models'),
              **kwargs):
    """
    Create MENet model with specific parameters.

    Parameters:
    ----------
    first_stage_channels : int
        Number of output channels at the first stage.
    side_channels : int
        Number of side channels in a ME-unit.
    groups : int
        Number of groups in convolution layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    layers = [4, 8, 4]

    if first_stage_channels == 108:
        init_block_channels = 12
        channels_per_layers = [108, 216, 432]
    elif first_stage_channels == 128:
        init_block_channels = 12
        channels_per_layers = [128, 256, 512]
    elif first_stage_channels == 160:
        init_block_channels = 16
        channels_per_layers = [160, 320, 640]
    elif first_stage_channels == 228:
        init_block_channels = 24
        channels_per_layers = [228, 456, 912]
    elif first_stage_channels == 256:
        init_block_channels = 24
        channels_per_layers = [256, 512, 1024]
    elif first_stage_channels == 348:
        init_block_channels = 24
        channels_per_layers = [348, 696, 1392]
    elif first_stage_channels == 352:
        init_block_channels = 24
        channels_per_layers = [352, 704, 1408]
    elif first_stage_channels == 456:
        init_block_channels = 48
        channels_per_layers = [456, 912, 1824]
    else:
        raise ValueError("The {} of `first_stage_channels` is not supported".format(first_stage_channels))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = menet(
        channels=channels,
        init_block_channels=init_block_channels,
        side_channels=side_channels,
        groups=groups,
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


def menet108_8x1_g3(**kwargs):
    """
    108-MENet-8x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=108, side_channels=8, groups=3, model_name="menet108_8x1_g3", **kwargs)


def menet128_8x1_g4(**kwargs):
    """
    128-MENet-8x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=128, side_channels=8, groups=4, model_name="menet128_8x1_g4", **kwargs)


def menet160_8x1_g8(**kwargs):
    """
    160-MENet-8x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=160, side_channels=8, groups=8, model_name="menet160_8x1_g8", **kwargs)


def menet228_12x1_g3(**kwargs):
    """
    228-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=228, side_channels=12, groups=3, model_name="menet228_12x1_g3", **kwargs)


def menet256_12x1_g4(**kwargs):
    """
    256-MENet-12x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=256, side_channels=12, groups=4, model_name="menet256_12x1_g4", **kwargs)


def menet348_12x1_g3(**kwargs):
    """
    348-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=348, side_channels=12, groups=3, model_name="menet348_12x1_g3", **kwargs)


def menet352_12x1_g8(**kwargs):
    """
    352-MENet-12x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=352, side_channels=12, groups=8, model_name="menet352_12x1_g8", **kwargs)


def menet456_24x1_g3(**kwargs):
    """
    456-MENet-24x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_menet(first_stage_channels=456, side_channels=24, groups=3, model_name="menet456_24x1_g3", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        menet108_8x1_g3,
        menet128_8x1_g4,
        menet160_8x1_g8,
        menet228_12x1_g3,
        menet256_12x1_g4,
        menet348_12x1_g3,
        menet352_12x1_g8,
        menet456_24x1_g3,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != menet108_8x1_g3 or weight_count == 654516)
        assert (model != menet128_8x1_g4 or weight_count == 750796)
        assert (model != menet160_8x1_g8 or weight_count == 850120)
        assert (model != menet228_12x1_g3 or weight_count == 1806568)
        assert (model != menet256_12x1_g4 or weight_count == 1888240)
        assert (model != menet348_12x1_g3 or weight_count == 3368128)
        assert (model != menet352_12x1_g8 or weight_count == 2272872)
        assert (model != menet456_24x1_g3 or weight_count == 5304784)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
