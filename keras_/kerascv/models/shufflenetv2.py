"""
    ShuffleNet V2 for ImageNet-1K, implemented in Keras.
    Original paper: 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.
"""

__all__ = ['shufflenetv2', 'shufflenetv2_wd2', 'shufflenetv2_w1', 'shufflenetv2_w3d2', 'shufflenetv2_w2']

import os
from keras import layers as nn
from keras.models import Model
from .common import (conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, maxpool2d, channel_shuffle_lambda,
                     se_block, batchnorm, is_channels_first, get_channel_axis, flatten)


def shuffle_unit(x,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 name="shuffle_unit"):
    """
    ShuffleNetV2 unit.

    Parameters
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_channels = out_channels // 2

    if downsample:
        y1 = depthwise_conv3x3(
            x=x,
            channels=in_channels,
            strides=2,
            name=name + "/dw_conv4")
        y1 = batchnorm(
            x=y1,
            name=name + "/dw_bn4")
        y1 = conv1x1(
            x=y1,
            in_channels=in_channels,
            out_channels=mid_channels,
            name=name + "/expand_conv5")
        y1 = batchnorm(
            x=y1,
            name=name + "/expand_bn5")
        y1 = nn.Activation("relu", name=name + "/expand_activ5")(y1)
        x2 = x
    else:
        in_split2_channels = in_channels // 2
        if is_channels_first():
            y1 = nn.Lambda(lambda z: z[:, 0:in_split2_channels, :, :])(x)
            x2 = nn.Lambda(lambda z: z[:, in_split2_channels:, :, :])(x)
        else:
            y1 = nn.Lambda(lambda z: z[:, :, :, 0:in_split2_channels])(x)
            x2 = nn.Lambda(lambda z: z[:, :, :, in_split2_channels:])(x)

    y2 = conv1x1(
        x=x2,
        in_channels=(in_channels if downsample else mid_channels),
        out_channels=mid_channels,
        name=name + "/compress_conv1")
    y2 = batchnorm(
        x=y2,
        name=name + "/compress_bn1")
    y2 = nn.Activation("relu", name=name + "/compress_activ1")(y2)

    y2 = depthwise_conv3x3(
        x=y2,
        channels=mid_channels,
        strides=(2 if downsample else 1),
        name=name + "/dw_conv2")
    y2 = batchnorm(
        x=y2,
        name=name + "/dw_bn2")

    y2 = conv1x1(
        x=y2,
        in_channels=mid_channels,
        out_channels=mid_channels,
        name=name + "/expand_conv3")
    y2 = batchnorm(
        x=y2,
        name=name + "/expand_bn3")
    y2 = nn.Activation("relu", name=name + "/expand_activ3")(y2)

    if use_se:
        y2 = se_block(
            x=y2,
            channels=mid_channels,
            name=name + "/se")

    if use_residual and not downsample:
        y2 = nn.add([y2, x2], name=name + "/add")

    x = nn.concatenate([y1, y2], axis=get_channel_axis(), name=name + "/concat")

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
    ShuffleNetV2 specific initial block.

    Parameters
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
        padding=0,
        ceil_mode=True,
        name=name + "/pool")
    return x


def shufflenetv2(channels,
                 init_block_channels,
                 final_block_channels,
                 use_se=False,
                 use_residual=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
    """
    ShuffleNetV2 model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    channels : list of list of int
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
    in_size : tuple of two ints, default (224, 224)
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


def get_shufflenetv2(width_scale,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".keras", "models"),
                     **kwargs):
    """
    Create ShuffleNetV2 model with specific parameters.

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

    init_block_channels = 24
    final_block_channels = 1024
    layers = [4, 8, 4]
    channels_per_layers = [116, 232, 464]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        if width_scale > 1.5:
            final_block_channels = int(final_block_channels * width_scale)

    net = shufflenetv2(
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


def shufflenetv2_wd2(**kwargs):
    """
    ShuffleNetV2 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2(width_scale=(12.0 / 29.0), model_name="shufflenetv2_wd2", **kwargs)


def shufflenetv2_w1(**kwargs):
    """
    ShuffleNetV2 1x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2(width_scale=1.0, model_name="shufflenetv2_w1", **kwargs)


def shufflenetv2_w3d2(**kwargs):
    """
    ShuffleNetV2 1.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2(width_scale=(44.0 / 29.0), model_name="shufflenetv2_w3d2", **kwargs)


def shufflenetv2_w2(**kwargs):
    """
    ShuffleNetV2 2x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2(width_scale=(61.0 / 29.0), model_name="shufflenetv2_w2", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        shufflenetv2_wd2,
        shufflenetv2_w1,
        shufflenetv2_w3d2,
        shufflenetv2_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shufflenetv2_wd2 or weight_count == 1366792)
        assert (model != shufflenetv2_w1 or weight_count == 2278604)
        assert (model != shufflenetv2_w3d2 or weight_count == 4406098)
        assert (model != shufflenetv2_w2 or weight_count == 7601686)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
