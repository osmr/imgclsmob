"""
    DarkNet, implemented in Keras.
    Original source: 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.
"""

__all__ = ['darknet', 'darknet_ref', 'darknet_tiny', 'darknet19']

import os
from keras import backend as K
from keras import layers as nn
from keras.models import Model
from .common import conv2d, GluonBatchNormalization


def dark_conv(x,
              in_channels,
              out_channels,
              kernel_size,
              padding,
              name="dark_conv"):
    """
    DarkNet specific convolution block.

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
    name : str, default 'dark_conv'
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
        use_bias=False,
        name=name + "/conv")
    x = GluonBatchNormalization(name=name + "/bn")(x)
    x = nn.LeakyReLU(alpha=0.1, name=name + "/activ")(x)
    return x


def dark_conv1x1(x,
                 in_channels,
                 out_channels,
                 name="dark_conv1x1"):
    """
    1x1 version of the DarkNet specific convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'dark_conv1x1'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return dark_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding=0,
        name=name)


def dark_conv3x3(x,
                 in_channels,
                 out_channels,
                 name="dark_conv3x3"):
    """
    3x3 version of the DarkNet specific convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'dark_conv3x3'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    return dark_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        name=name)


def dark_convYxY(x,
                 in_channels,
                 out_channels,
                 pointwise=True,
                 name="dark_convYxY"):
    """
    DarkNet unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    pointwise : bool
        Whether use 1x1 (pointwise) convolution or 3x3 convolution.
    name : str, default 'dark_convYxY'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    if pointwise:
        return dark_conv1x1(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
    else:
        return dark_conv3x3(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)


def darknet(channels,
            odd_pointwise,
            avg_pool_size,
            cls_activ,
            in_channels=3,
            classes=1000):
    """
    DarkNet model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    odd_pointwise : bool
        Whether pointwise convolution layer is used for each odd unit.
    avg_pool_size : int
        Window size of the final average pooling.
    cls_activ : bool
        Whether classification convolution layer uses an activation.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if K.image_data_format() == 'channels_first' else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = input
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            x = dark_convYxY(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                pointwise=(len(channels_per_stage) > 1) and not(((j + 1) % 2 == 1) ^ odd_pointwise),
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
        if i != len(channels) - 1:
            x = nn.MaxPool2D(
                pool_size=2,
                strides=2,
                name="features/pool{}".format(i + 1))(x)

    x = nn.Conv2D(
        filters=classes,
        kernel_size=1,
        name="output/final_conv")(x)
    if cls_activ:
        x = nn.LeakyReLU(alpha=0.1, name="output/final_activ")(x)
    x = nn.AvgPool2D(
        pool_size=avg_pool_size,
        strides=1,
        name="output/final_pool")(x)
    x = nn.Flatten()(x)

    model = Model(inputs=input, outputs=x)
    return model


def get_darknet(version,
                model_name=None,
                pretrained=False,
                root=os.path.join('~', '.keras', 'models'),
                **kwargs):
    """
    Create DarkNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('ref', 'tiny' or '19').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    if version == 'ref':
        channels = [[16], [32], [64], [128], [256], [512], [1024]]
        odd_pointwise = False
        avg_pool_size = 3
        cls_activ = True
    elif version == 'tiny':
        channels = [[16], [32], [16, 128, 16, 128], [32, 256, 32, 256], [64, 512, 64, 512, 128]]
        odd_pointwise = True
        avg_pool_size = 14
        cls_activ = False
    elif version == '19':
        channels = [[32], [64], [128, 64, 128], [256, 128, 256], [512, 256, 512, 256, 512],
                    [1024, 512, 1024, 512, 1024]]
        odd_pointwise = False
        avg_pool_size = 7
        cls_activ = False
    # elif version == '53':
    #     init_block_channels = 32
    #     layers = [2, 8, 8, 4]
    #     channels_per_layers = [64, 128, 256, 512, 1024]
    #     channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    #     odd_pointwise = False
    #     avg_pool_size = 7
    #     cls_activ = False
    else:
        raise ValueError("Unsupported DarkNet version {}".format(version))

    net = darknet(
        channels=channels,
        odd_pointwise=odd_pointwise,
        avg_pool_size=avg_pool_size,
        cls_activ=cls_activ,
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


def darknet_ref(**kwargs):
    """
    DarkNet 'Reference' model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_darknet(version="ref", model_name="darknet_ref", **kwargs)


def darknet_tiny(**kwargs):
    """
    DarkNet Tiny model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_darknet(version="tiny", model_name="darknet_tiny", **kwargs)


def darknet19(**kwargs):
    """
    DarkNet-19 model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_darknet(version="19", model_name="darknet19", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False
    keras.backend.set_learning_phase(0)

    models = [
        darknet_ref,
        darknet_tiny,
        darknet19,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darknet_ref or weight_count == 7319416)
        assert (model != darknet_tiny or weight_count == 1042104)
        assert (model != darknet19 or weight_count == 20842376)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
