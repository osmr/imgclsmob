"""
    SqueezeNet, implemented in Keras.
    Original paper: 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,'
    https://arxiv.org/abs/1602.07360.
"""

__all__ = ['squeezenet', 'squeezenet_v1_0', 'squeezenet_v1_1', 'squeezeresnet_v1_0', 'squeezeresnet_v1_1']

import os
from keras import layers as nn
from keras.models import Model
from .common import max_pool2d_ceil, conv2d, is_channels_first, get_channel_axis, flatten


def fire_conv(x,
              in_channels,
              out_channels,
              kernel_size,
              padding,
              name="fire_conv"):
    """
    SqueezeNet specific convolution block.

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
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    name : str, default 'fire_conv'
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
        kernel_size=kernel_size,
        padding=padding,
        use_bias=True,
        name=name + "/conv")
    x = nn.Activation("relu", name=name + "/activ")(x)
    return x


def fire_unit(x,
              in_channels,
              squeeze_channels,
              expand1x1_channels,
              expand3x3_channels,
              residual,
              name="fire_unit"):
    """
    SqueezeNet unit, so-called 'Fire' unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    squeeze_channels : int
        Number of output channels for squeeze convolution blocks.
    expand1x1_channels : int
        Number of output channels for expand 1x1 convolution blocks.
    expand3x3_channels : int
        Number of output channels for expand 3x3 convolution blocks.
    residual : bool
        Whether use residual connection.
    name : str, default 'fire_unit'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if residual:
        identity = x

    x = fire_conv(
        x=x,
        in_channels=in_channels,
        out_channels=squeeze_channels,
        kernel_size=1,
        padding=0,
        name=name + "/squeeze")
    y1 = fire_conv(
        x=x,
        in_channels=squeeze_channels,
        out_channels=expand1x1_channels,
        kernel_size=1,
        padding=0,
        name=name + "/expand1x1")
    y2 = fire_conv(
        x=x,
        in_channels=squeeze_channels,
        out_channels=expand3x3_channels,
        kernel_size=3,
        padding=1,
        name=name + "/expand3x3")

    out = nn.concatenate([y1, y2], axis=get_channel_axis(), name=name + "/concat")

    if residual:
        out = nn.add([out, identity], name=name + "/add")

    return out


def squeeze_init_block(x,
                       in_channels,
                       out_channels,
                       kernel_size,
                       name="squeeze_init_block"):
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
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    name : str, default 'squeeze_init_block'
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
        kernel_size=kernel_size,
        strides=2,
        use_bias=True,
        name=name + "/conv")
    x = nn.Activation("relu", name=name + "/activ")(x)
    return x


def squeezenet(channels,
               residuals,
               init_block_kernel_size,
               init_block_channels,
               in_channels=3,
               in_size=(224, 224),
               classes=1000):
    """
    SqueezeNet model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,'
    https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    residuals : bool
        Whether to use residual units.
    init_block_kernel_size : int or tuple/list of 2 int
        The dimensions of the convolution window for the initial unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if is_channels_first() else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = squeeze_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        kernel_size=init_block_kernel_size,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        x = max_pool2d_ceil(
            x=x,
            pool_size=3,
            strides=2,
            padding="valid",
            name="features/pool{}".format(i + 1))
        for j, out_channels in enumerate(channels_per_stage):
            expand_channels = out_channels // 2
            squeeze_channels = out_channels // 8
            x = fire_unit(
                x=x,
                in_channels=in_channels,
                squeeze_channels=squeeze_channels,
                expand1x1_channels=expand_channels,
                expand3x3_channels=expand_channels,
                residual=((residuals is not None) and (residuals[i][j] == 1)),
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = nn.Dropout(
        rate=0.5,
        name="features/dropout")(x)

    x = nn.Conv2D(
        filters=classes,
        kernel_size=1,
        name="output/final_conv")(x)
    x = nn.Activation("relu", name="output/final_activ")(x)
    x = nn.AvgPool2D(
        pool_size=13,
        strides=1,
        name="output/final_pool")(x)
    # x = nn.Flatten()(x)
    x = flatten(x)

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_squeezenet(version,
                   residual=False,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join('~', '.keras', 'models'),
                   **kwargs):
    """
    Create SqueezeNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('1.0' or '1.1').
    residual : bool, default False
        Whether to use residual connections.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    if version == '1.0':
        channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
        residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]
        init_block_kernel_size = 7
        init_block_channels = 96
    elif version == '1.1':
        channels = [[128, 128], [256, 256], [384, 384, 512, 512]]
        residuals = [[0, 1], [0, 1], [0, 1, 0, 1]]
        init_block_kernel_size = 3
        init_block_channels = 64
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    if not residual:
        residuals = None

    net = squeezenet(
        channels=channels,
        residuals=residuals,
        init_block_kernel_size=init_block_kernel_size,
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


def squeezenet_v1_0(**kwargs):
    """
    SqueezeNet 'vanilla' model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
    size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.0", residual=False, model_name="squeezenet_v1_0", **kwargs)


def squeezenet_v1_1(**kwargs):
    """
    SqueezeNet v1.1 model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
    size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.1", residual=False, model_name="squeezenet_v1_1", **kwargs)


def squeezeresnet_v1_0(**kwargs):
    """
    SqueezeNet model with residual connections from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
    <0.5MB model size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.0", residual=True, model_name="squeezeresnet_v1_0", **kwargs)


def squeezeresnet_v1_1(**kwargs):
    """
    SqueezeNet v1.1 model with residual connections from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size,' https://arxiv.org/abs/1602.07360.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet(version="1.1", residual=True, model_name="squeezeresnet_v1_1", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        squeezenet_v1_0,
        squeezenet_v1_1,
        squeezeresnet_v1_0,
        squeezeresnet_v1_1,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != squeezenet_v1_0 or weight_count == 1248424)
        assert (model != squeezenet_v1_1 or weight_count == 1235496)
        assert (model != squeezeresnet_v1_0 or weight_count == 1248424)
        assert (model != squeezeresnet_v1_1 or weight_count == 1235496)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
