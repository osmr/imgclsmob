"""
    VGG for ImageNet-1K, implemented in Keras.
    Original paper: 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.
"""

__all__ = ['vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'bn_vgg11', 'bn_vgg13', 'bn_vgg16', 'bn_vgg19', 'bn_vgg11b',
           'bn_vgg13b', 'bn_vgg16b', 'bn_vgg19b']

import os
from keras import layers as nn
from keras.models import Model
from .common import conv3x3_block, is_channels_first, flatten


def vgg_dense(x,
              in_channels,
              out_channels,
              name="vgg_dense"):
    """
    VGG specific dense block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'vgg_dense'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = nn.Dense(
        units=out_channels,
        input_dim=in_channels,
        name=name + "/fc")(x)
    x = nn.Activation("relu", name=name + "/activ")(x)
    x = nn.Dropout(
        rate=0.5,
        name=name + "/dropout")(x)
    return x


def vgg_output_block(x,
                     in_channels,
                     classes,
                     name="vgg_output_block"):
    """
    VGG specific output block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    name : str, default 'vgg_output_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_channels = 4096

    x = vgg_dense(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        name=name + "/fc1")
    x = vgg_dense(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        name=name + "/fc2")
    x = nn.Dense(
        units=classes,
        input_dim=mid_channels,
        name=name + "/fc3")(x)
    return x


def vgg(channels,
        use_bias=True,
        use_bn=False,
        in_channels=3,
        in_size=(224, 224),
        classes=1000):
    """
    VGG models from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default False
        Whether to use BatchNorm layers.
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

    x = input
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            x = conv3x3_block(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
        x = nn.MaxPool2D(
            pool_size=2,
            strides=2,
            padding="valid",
            name="features/stage{}/pool".format(i + 1))(x)

    x = flatten(x, reshape=True)
    x = vgg_output_block(
        x=x,
        in_channels=(in_channels * 7 * 7),
        classes=classes,
        name="output")

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_vgg(blocks,
            use_bias=True,
            use_bn=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".keras", "models"),
            **kwargs):
    """
    Create VGG model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default False
        Whether to use BatchNorm layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    if blocks == 11:
        layers = [1, 1, 2, 2, 2]
    elif blocks == 13:
        layers = [2, 2, 2, 2, 2]
    elif blocks == 16:
        layers = [2, 2, 3, 3, 3]
    elif blocks == 19:
        layers = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Unsupported VGG with number of blocks: {}".format(blocks))

    channels_per_layers = [64, 128, 256, 512, 512]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = vgg(
        channels=channels,
        use_bias=use_bias,
        use_bn=use_bn,
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


def vgg11(**kwargs):
    """
    VGG-11 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=11, model_name="vgg11", **kwargs)


def vgg13(**kwargs):
    """
    VGG-13 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=13, model_name="vgg13", **kwargs)


def vgg16(**kwargs):
    """
    VGG-16 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=16, model_name="vgg16", **kwargs)


def vgg19(**kwargs):
    """
    VGG-19 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=19, model_name="vgg19", **kwargs)


def bn_vgg11(**kwargs):
    """
    VGG-11 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=11, use_bias=False, use_bn=True, model_name="bn_vgg11", **kwargs)


def bn_vgg13(**kwargs):
    """
    VGG-13 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=13, use_bias=False, use_bn=True, model_name="bn_vgg13", **kwargs)


def bn_vgg16(**kwargs):
    """
    VGG-16 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=16, use_bias=False, use_bn=True, model_name="bn_vgg16", **kwargs)


def bn_vgg19(**kwargs):
    """
    VGG-19 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=19, use_bias=False, use_bn=True, model_name="bn_vgg19", **kwargs)


def bn_vgg11b(**kwargs):
    """
    VGG-11 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=11, use_bias=True, use_bn=True, model_name="bn_vgg11b", **kwargs)


def bn_vgg13b(**kwargs):
    """
    VGG-13 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=13, use_bias=True, use_bn=True, model_name="bn_vgg13b", **kwargs)


def bn_vgg16b(**kwargs):
    """
    VGG-16 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=16, use_bias=True, use_bn=True, model_name="bn_vgg16b", **kwargs)


def bn_vgg19b(**kwargs):
    """
    VGG-19 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=19, use_bias=True, use_bn=True, model_name="bn_vgg19b", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        vgg11,
        vgg13,
        vgg16,
        vgg19,
        bn_vgg11,
        bn_vgg13,
        bn_vgg16,
        bn_vgg19,
        bn_vgg11b,
        bn_vgg13b,
        bn_vgg16b,
        bn_vgg19b,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != vgg11 or weight_count == 132863336)
        assert (model != vgg13 or weight_count == 133047848)
        assert (model != vgg16 or weight_count == 138357544)
        assert (model != vgg19 or weight_count == 143667240)
        assert (model != bn_vgg11 or weight_count == 132866088)
        assert (model != bn_vgg13 or weight_count == 133050792)
        assert (model != bn_vgg16 or weight_count == 138361768)
        assert (model != bn_vgg19 or weight_count == 143672744)
        assert (model != bn_vgg11b or weight_count == 132868840)
        assert (model != bn_vgg13b or weight_count == 133053736)
        assert (model != bn_vgg16b or weight_count == 138365992)
        assert (model != bn_vgg19b or weight_count == 143678248)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
