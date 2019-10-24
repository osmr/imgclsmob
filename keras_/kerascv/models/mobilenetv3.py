"""
    MobileNetV3 for ImageNet-1K, implemented in Keras.
    Original paper: 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
"""

__all__ = ['mobilenetv3', 'mobilenetv3_small_w7d20', 'mobilenetv3_small_wd2', 'mobilenetv3_small_w3d4',
           'mobilenetv3_small_w1', 'mobilenetv3_small_w5d4', 'mobilenetv3_large_w7d20', 'mobilenetv3_large_wd2',
           'mobilenetv3_large_w3d4', 'mobilenetv3_large_w1', 'mobilenetv3_large_w5d4']

import os
from keras import layers as nn
from keras.models import Model
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block,\
    se_block, HSwish, is_channels_first, flatten


def mobilenetv3_unit(x,
                     in_channels,
                     out_channels,
                     exp_channels,
                     strides,
                     use_kernel3,
                     activation,
                     use_se,
                     name="mobilenetv3_unit"):
    """
    MobileNetV3 unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    exp_channels : int
        Number of middle (expanded) channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    activation : str
        Activation function or name of activation function.
    use_se : bool
        Whether to use SE-module.
    name : str, default 'mobilenetv3_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    assert (exp_channels >= out_channels)
    residual = (in_channels == out_channels) and (strides == 1)
    use_exp_conv = exp_channels != out_channels
    mid_channels = exp_channels

    if residual:
        identity = x

    if use_exp_conv:
        x = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=activation,
            name=name + "/exp_conv")
    if use_kernel3:
        x = dwconv3x3_block(
            x=x,
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            activation=activation,
            name=name + "/conv1")
    else:
        x = dwconv5x5_block(
            x=x,
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            activation=activation,
            name=name + "/conv1")
    if use_se:
        x = se_block(
            x=x,
            channels=mid_channels,
            reduction=4,
            approx_sigmoid=True,
            round_mid=True,
            name=name + "/se")
    x = conv1x1_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activation=None,
        name=name + "/conv2")

    if residual:
        x = nn.add([x, identity], name=name + "/add")

    return x


def mobilenetv3_final_block(x,
                            in_channels,
                            out_channels,
                            use_se,
                            name="mobilenetv3_final_block"):
    """
    MobileNetV3 final block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_se : bool
        Whether to use SE-module.
    name : str, default 'mobilenetv3_final_block'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        activation="hswish",
        name=name + "/conv")
    if use_se:
        x = se_block(
            x=x,
            channels=out_channels,
            reduction=4,
            approx_sigmoid=True,
            round_mid=True,
            name=name + "/se")
    return x


def mobilenetv3_classifier(x,
                           in_channels,
                           out_channels,
                           mid_channels,
                           dropout_rate,
                           name="mobilenetv3_final_block"):
    """
    MobileNetV3 classifier.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    name : str, default 'mobilenetv3_classifier'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        name=name + "/conv1")
    x = HSwish(name=name + "/hswish")(x)

    use_dropout = (dropout_rate != 0.0)
    if use_dropout:
        x = nn.Dropout(
            rate=dropout_rate,
            name=name + "dropout")(x)

    x = conv1x1(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        use_bias=True,
        name=name + "/conv2")
    return x


def mobilenetv3(channels,
                exp_channels,
                init_block_channels,
                final_block_channels,
                classifier_mid_channels,
                kernels3,
                use_relu,
                use_se,
                first_stride,
                final_use_se,
                in_channels=3,
                in_size=(224, 224),
                classes=1000):
    """
    MobileNetV3 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    exp_channels : list of list of int
        Number of middle (expanded) channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    use_relu : list of list of int/bool
        Using ReLU activation flag for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
    final_use_se : bool
        Whether to use SE-module in the final block.
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
        activation="hswish",
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            exp_channels_ij = exp_channels[i][j]
            strides = 2 if (j == 0) and ((i != 0) or first_stride) else 1
            use_kernel3 = kernels3[i][j] == 1
            activation = "relu" if use_relu[i][j] == 1 else "hswish"
            use_se_flag = use_se[i][j] == 1
            x = mobilenetv3_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                exp_channels=exp_channels_ij,
                use_kernel3=use_kernel3,
                strides=strides,
                activation=activation,
                use_se=use_se_flag,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = mobilenetv3_final_block(
        x=x,
        in_channels=in_channels,
        out_channels=final_block_channels,
        use_se=final_use_se,
        name="features/final_block")
    in_channels = final_block_channels
    x = nn.AvgPool2D(
        pool_size=7,
        strides=1,
        name="features/final_pool")(x)

    x = mobilenetv3_classifier(
        x=x,
        in_channels=in_channels,
        out_channels=classes,
        mid_channels=classifier_mid_channels,
        dropout_rate=0.2,
        name="output")
    x = flatten(x)

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_mobilenetv3(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".keras", "models"),
                    **kwargs):
    """
    Create MobileNetV3 model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('small' or 'large').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    if version == "small":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        first_stride = True
        final_block_channels = 576
    elif version == "large":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        first_stride = False
        final_block_channels = 960
    else:
        raise ValueError("Unsupported MobileNetV3 version {}".format(version))

    final_use_se = False
    classifier_mid_channels = 1280

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        exp_channels = [[round_channels(cij * width_scale) for cij in ci] for ci in exp_channels]
        init_block_channels = round_channels(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale)

    net = mobilenetv3(
        channels=channels,
        exp_channels=exp_channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        use_relu=use_relu,
        use_se=use_se,
        first_stride=first_stride,
        final_use_se=final_use_se,
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


def mobilenetv3_small_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_small_wd2(**kwargs):
    """
    MobileNetV3 Small 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.5, model_name="mobilenetv3_small_wd2", **kwargs)


def mobilenetv3_small_w3d4(**kwargs):
    """
    MobileNetV3 Small 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.75, model_name="mobilenetv3_small_w3d4", **kwargs)


def mobilenetv3_small_w1(**kwargs):
    """
    MobileNetV3 Small 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.0, model_name="mobilenetv3_small_w1", **kwargs)


def mobilenetv3_small_w5d4(**kwargs):
    """
    MobileNetV3 Small 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.25, model_name="mobilenetv3_small_w5d4", **kwargs)


def mobilenetv3_large_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_large_wd2(**kwargs):
    """
    MobileNetV3 Large 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.5, model_name="mobilenetv3_large_wd2", **kwargs)


def mobilenetv3_large_w3d4(**kwargs):
    """
    MobileNetV3 Large 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.75, model_name="mobilenetv3_large_w3d4", **kwargs)


def mobilenetv3_large_w1(**kwargs):
    """
    MobileNetV3 Large 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.0, model_name="mobilenetv3_large_w1", **kwargs)


def mobilenetv3_large_w5d4(**kwargs):
    """
    MobileNetV3 Large 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.25, model_name="mobilenetv3_large_w5d4", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        mobilenetv3_small_w7d20,
        mobilenetv3_small_wd2,
        mobilenetv3_small_w3d4,
        mobilenetv3_small_w1,
        mobilenetv3_small_w5d4,
        mobilenetv3_large_w7d20,
        mobilenetv3_large_wd2,
        mobilenetv3_large_w3d4,
        mobilenetv3_large_w1,
        mobilenetv3_large_w5d4,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenetv3_small_w7d20 or weight_count == 2159600)
        assert (model != mobilenetv3_small_wd2 or weight_count == 2288976)
        assert (model != mobilenetv3_small_w3d4 or weight_count == 2581312)
        assert (model != mobilenetv3_small_w1 or weight_count == 2945288)
        assert (model != mobilenetv3_small_w5d4 or weight_count == 3643632)
        assert (model != mobilenetv3_large_w7d20 or weight_count == 2943080)
        assert (model != mobilenetv3_large_wd2 or weight_count == 3334896)
        assert (model != mobilenetv3_large_w3d4 or weight_count == 4263496)
        assert (model != mobilenetv3_large_w1 or weight_count == 5481752)
        assert (model != mobilenetv3_large_w5d4 or weight_count == 7459144)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
