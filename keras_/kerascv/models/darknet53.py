"""
    DarkNet-53, implemented in Keras.
    Original source: 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.
"""

__all__ = ['darknet53_model', 'darknet53']

import os
from keras import backend as K
from keras import layers as nn
from keras.models import Model
from .common import conv1x1_block, conv3x3_block


def dark_unit(x,
              in_channels,
              out_channels,
              alpha,
              name="dark_unit"):
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
    alpha : float
        Slope coefficient for Leaky ReLU activation.
    name : str, default 'dark_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    assert (out_channels % 2 == 0)
    mid_channels = out_channels // 2

    identity = x
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        activation=nn.LeakyReLU(
            alpha=alpha,
            name=name + "/conv1/activ"),
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activation=nn.LeakyReLU(
            alpha=alpha,
            name=name + "/conv2/activ"),
        name=name + "/conv2")
    x = nn.add([x, identity], name=name + "/add")
    return x


def darknet53_model(channels,
                    init_block_channels,
                    alpha=0.1,
                    in_channels=3,
                    in_size=(224, 224),
                    classes=1000):
    """
    DarkNet-53 model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if K.image_data_format() == "channels_first" else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = conv3x3_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        activation=nn.LeakyReLU(
            alpha=alpha,
            name="features/init_block/activ"),
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            if j == 0:
                x = conv3x3_block(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=2,
                    activation=nn.LeakyReLU(
                        alpha=alpha,
                        name="features/stage{}/unit{}/active".format(i + 1, j + 1)),
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
            else:
                x = dark_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    alpha=alpha,
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
    model.in_size = in_size
    model.classes = classes
    return model


def get_darknet53(model_name=None,
                  pretrained=False,
                  root=os.path.join('~', '.keras', 'models'),
                  **kwargs):
    """
    Create DarkNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [2, 3, 9, 9, 5]
    channels_per_layers = [64, 128, 256, 512, 1024]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = darknet53_model(
        channels=channels,
        init_block_channels=init_block_channels,
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


def darknet53(**kwargs):
    """
    DarkNet-53 'Reference' model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_darknet53(model_name="darknet53", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        darknet53,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darknet53 or weight_count == 41609928)

        if K.image_data_format() == 'channels_first':
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
