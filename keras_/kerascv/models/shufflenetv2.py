"""
    ShuffleNet V2, implemented in Keras.
    Original paper: 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.
"""

__all__ = ['shufflenetv2', 'shufflenetv2_wd2', 'shufflenetv2_w1', 'shufflenetv2_w3d2', 'shufflenetv2_w2']

import os
from keras import backend as K
from keras import layers as nn
from keras.models import Model
from .common import conv2d, conv1x1, max_pool2d_ceil, channel_shuffle_lambda, se_block, GluonBatchNormalization


def shuffle_conv(x,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 name="shuffle_conv"):
    """
    ShuffleNetV2 specific convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    name : str, default 'shuffle_conv'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name + "/conv")
    x = GluonBatchNormalization(name=name + "/bn")(x)
    x = nn.Activation("relu", name=name + "/activ")(x)
    return x


def shuffle_conv1x1(x,
                  in_channels,
                  out_channels,
                  name="shuffle_conv1x1"):
    """
    1x1 version of the ShuffleNetV2 specific convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'shuffle_conv1x1'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return shuffle_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        name=name)


def depthwise_conv3x3(x,
                      channels,
                      strides,
                      name="depthwise_conv3x3"):
    """
    Depthwise convolution 3x3 layer.

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


def shuffle_unit(x,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 name="shuffle_unit"):
    """
    ShuffleNetV2 unit.

    Parameters:
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
        y1 = GluonBatchNormalization(name=name + "/dw_bn4")(y1)
        y1 = conv1x1(
            out_channels=mid_channels,
            name=name + "/expand_conv5")(y1)
        y1 = GluonBatchNormalization(name=name + "/expand_bn5")(y1)
        y1 = nn.Activation("relu", name=name + "/expand_activ5")(y1)
        x2 = x
    else:
        in_split2_channels = in_channels // 2
        y1 = nn.Lambda(lambda z: z[:, 0:in_split2_channels, :, :])(x)
        x2 = nn.Lambda(lambda z: z[:, in_split2_channels:, :, :])(x)

    y2 = conv1x1(
        out_channels=mid_channels,
        name=name + "/compress_conv1")(x2)
    y2 = GluonBatchNormalization(name=name + "/compress_bn1")(y2)
    y2 = nn.Activation("relu", name=name + "/compress_activ1")(y2)

    y2 = depthwise_conv3x3(
        x=y2,
        channels=mid_channels,
        strides=(2 if downsample else 1),
        name=name + "/dw_conv2")
    y2 = GluonBatchNormalization(name=name + "/dw_bn2")(y2)

    y2 = conv1x1(
        out_channels=mid_channels,
        name=name + "/expand_conv3")(y2)
    y2 = GluonBatchNormalization(name=name + "/expand_bn3")(y2)
    y2 = nn.Activation("relu", name=name + "/compress_activ3")(y2)

    if use_se:
        y2 = se_block(
            x=y2,
            channels=out_channels,
            name=name + "/se")

    if use_residual and not downsample:
        y2 = nn.add([y2, x2], name=name + "/add")

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = nn.concatenate([y1, y2], axis=channel_axis, name=name + "/concat")

    x = channel_shuffle_lambda(
        channels=mid_channels,
        groups=2,
        name=name + "/c_shuffle")(x)

    return x


def shuffle_init_block(x,
                       in_channels,
                       out_channels,
                       name="shuffle_init_block"):
    """
    ShuffleNetV2 specific initial block.

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
    x = shuffle_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=2,
        padding=1,
        name=name + "/conv")
    x = max_pool2d_ceil(
        x=x,
        pool_size=3,
        strides=2,
        padding="valid",
        name=name + "/pool")
    return x


def shufflenetv2(channels,
                 init_block_channels,
                 final_block_channels,
                 use_se=False,
                 use_residual=False,
                 in_channels=3,
                 classes=1000):
    """
    ShuffleNetV2 model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
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
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if K.image_data_format() == 'channels_first' else (224, 224, in_channels)
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
    x = shuffle_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=final_block_channels,
        name="features/final_block")
    in_channels = final_block_channels
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


def get_shufflenetv2(width_scale,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join('~', '.keras', 'models'),
                     **kwargs):
    """
    Create ShuffleNetV2 model with specific parameters.

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
        from .model_store import get_model_file
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def shufflenetv2_wd2(**kwargs):
    """
    ShuffleNetV2 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
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

    Parameters:
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

    Parameters:
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

    Parameters:
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

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
