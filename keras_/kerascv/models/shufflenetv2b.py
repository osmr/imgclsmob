"""
    ShuffleNet V2 for ImageNet-1K, implemented in Keras. The alternative variant.
    Original paper: 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.
"""

__all__ = ['shufflenetv2b', 'shufflenetv2b_wd2', 'shufflenetv2b_w1', 'shufflenetv2b_w3d2', 'shufflenetv2b_w2']

import os
from keras import layers as nn
from keras.models import Model
from .common import (conv1x1_block, conv3x3_block, dwconv3x3_block, channel_shuffle_lambda, maxpool2d, se_block,
                     is_channels_first, get_channel_axis, flatten)


def shuffle_unit(x,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 name="shuffle_unit"):
    """
    ShuffleNetV2(b) unit.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether do downsample.
    use_se : bool
        Whether to use SE block.
    use_residual : bool
        Whether to use residual connection.
    name : str, default 'shuffle_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 2
    in_channels2 = in_channels // 2
    assert (in_channels % 2 == 0)

    if downsample:
        y1 = dwconv3x3_block(
            x=x,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            activation=None,
            name=name + "/shortcut_dconv")
        y1 = conv1x1_block(
            x=y1,
            in_channels=in_channels,
            out_channels=in_channels,
            name=name + "/shortcut_conv")
        x2 = x
    else:
        if is_channels_first():
            y1 = nn.Lambda(lambda z: z[:, 0:in_channels2, :, :])(x)
            x2 = nn.Lambda(lambda z: z[:, in_channels2:, :, :])(x)
        else:
            y1 = nn.Lambda(lambda z: z[:, :, :, 0:in_channels2])(x)
            x2 = nn.Lambda(lambda z: z[:, :, :, in_channels2:])(x)

    y2_in_channels = (in_channels if downsample else in_channels2)
    y2_out_channels = out_channels - y2_in_channels

    y2 = conv1x1_block(
        x=x2,
        in_channels=y2_in_channels,
        out_channels=mid_channels,
        name=name + "/conv1")
    y2 = dwconv3x3_block(
        x=y2,
        in_channels=mid_channels,
        out_channels=mid_channels,
        strides=(2 if downsample else 1),
        activation=None,
        name=name + "/dconv")
    y2 = conv1x1_block(
        x=y2,
        in_channels=mid_channels,
        out_channels=y2_out_channels,
        name=name + "/conv2")

    if use_se:
        y2 = se_block(
            x=y2,
            channels=y2_out_channels,
            name=name + "/se")

    if use_residual and not downsample:
        assert (y2_out_channels == in_channels2)
        y2 = nn.add([y2, x2], name=name + "/add")

    x = nn.concatenate([y1, y2], axis=get_channel_axis(), name=name + "/concat")

    assert (out_channels % 2 == 0)
    x = channel_shuffle_lambda(
        channels=out_channels,
        groups=2,
        name=name + "/c_shuffle")(x)

    return x


def shuffle_init_block(x,
                       in_channels,
                       out_channels,
                       name="shuffle_init_block"):
    """
    ShuffleNetV2(b) specific initial block.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'shuffle_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        name=name + "/conv")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=1,
        ceil_mode=False,
        name=name + "/pool")
    return x


def shufflenetv2b(channels,
                  init_block_channels,
                  final_block_channels,
                  use_se=False,
                  use_residual=False,
                  in_channels=3,
                  in_size=(224, 224),
                  classes=1000):
    """
    ShuffleNetV2(b) model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    use_se : bool, default False
        Whether to use SE block.
    use_residual : bool, default False
        Whether to use residual connections.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, in_size[0], in_size[1]) if is_channels_first() else\
        (in_size[0], in_size[1], in_channels)
    input = nn.Input(shape=input_shape)

    x = shuffle_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            downsample = (j == 0)
            x = shuffle_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=downsample,
                use_se=use_se,
                use_residual=use_residual,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=final_block_channels,
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


def get_shufflenetv2b(width_scale,
                      model_name=None,
                      pretrained=False,
                      root=os.path.join("~", ".keras", "models"),
                      **kwargs):
    """
    Create ShuffleNetV2(b) model with specific parameters.

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

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """

    init_block_channels = 24
    final_block_channels = 1024
    layers = [4, 8, 4]
    channels_per_layers = [116, 232, 464]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        if width_scale > 1.5:
            final_block_channels = int(final_block_channels * width_scale)

    net = shufflenetv2b(
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


def shufflenetv2b_wd2(**kwargs):
    """
    ShuffleNetV2(b) 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenetv2b(
        width_scale=(12.0 / 29.0),
        model_name="shufflenetv2b_wd2",
        **kwargs)


def shufflenetv2b_w1(**kwargs):
    """
    ShuffleNetV2(b) 1x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenetv2b(
        width_scale=1.0,
        model_name="shufflenetv2b_w1",
        **kwargs)


def shufflenetv2b_w3d2(**kwargs):
    """
    ShuffleNetV2(b) 1.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenetv2b(
        width_scale=(44.0 / 29.0),
        model_name="shufflenetv2b_w3d2",
        **kwargs)


def shufflenetv2b_w2(**kwargs):
    """
    ShuffleNetV2(b) 2x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenetv2b(
        width_scale=(61.0 / 29.0),
        model_name="shufflenetv2b_w2",
        **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        shufflenetv2b_wd2,
        shufflenetv2b_w1,
        shufflenetv2b_w3d2,
        shufflenetv2b_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shufflenetv2b_wd2 or weight_count == 1366792)
        assert (model != shufflenetv2b_w1 or weight_count == 2279760)
        assert (model != shufflenetv2b_w3d2 or weight_count == 4410194)
        assert (model != shufflenetv2b_w2 or weight_count == 7611290)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
