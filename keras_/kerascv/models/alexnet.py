"""
    AlexNet for ImageNet-1K, implemented in Keras.
    Original paper: 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.
"""

__all__ = ['alexnet_model', 'alexnet', 'alexnetb']

import os
from keras import layers as nn
from keras.models import Model
from .common import conv_block, maxpool2d, is_channels_first, flatten, lrn


def alex_conv(x,
              in_channels,
              out_channels,
              kernel_size,
              strides,
              padding,
              use_lrn,
              name="alex_conv"):
    """
    AlexNet specific convolution block.

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
    use_lrn : bool
        Whether to use LRN layer.
    name : str, default 'alex_conv'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = conv_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=True,
        use_bn=False,
        name=name + "/conv")
    if use_lrn:
        x = lrn(x)
    return x


def alex_dense(x,
               in_channels,
               out_channels,
               name="alex_dense"):
    """
    AlexNet specific dense block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'alex_dense'
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


def alex_output_block(x,
                      in_channels,
                      classes,
                      name="alex_output_block"):
    """
    AlexNet specific output block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    name : str, default 'alex_output_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    mid_channels = 4096

    x = alex_dense(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        name=name + "/fc1")
    x = alex_dense(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        name=name + "/fc2")
    x = nn.Dense(
        units=classes,
        input_dim=mid_channels,
        name=name + "/fc3")(x)
    return x


def alexnet_model(channels,
                  kernel_sizes,
                  strides,
                  paddings,
                  use_lrn,
                  in_channels=3,
                  in_size=(224, 224),
                  classes=1000):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    kernel_sizes : list of list of int
        Convolution window sizes for each unit.
    strides : list of list of int or tuple/list of 2 int
        Strides of the convolution for each unit.
    paddings : list of list of int or tuple/list of 2 int
        Padding value for convolution layer for each unit.
    use_lrn : bool
        Whether to use LRN layer.
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
        use_lrn_i = use_lrn and (i in [0, 1])
        for j, out_channels in enumerate(channels_per_stage):
            x = alex_conv(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i][j],
                strides=strides[i][j],
                padding=paddings[i][j],
                use_lrn=use_lrn_i,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
        x = maxpool2d(
            x=x,
            pool_size=3,
            strides=2,
            padding=0,
            ceil_mode=True,
            name="features/stage{}/pool".format(i + 1))

    x = flatten(x, reshape=True)
    x = alex_output_block(
        x=x,
        in_channels=(in_channels * 6 * 6),
        classes=classes,
        name="output")

    model = Model(inputs=input, outputs=x)
    model.in_size = in_size
    model.classes = classes
    return model


def get_alexnet(version="a",
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".keras", "models"),
                **kwargs):
    """
    Create AlexNet model with specific parameters.

    Parameters:
    ----------
    version : str, default 'a'
        Version of AlexNet ('a' or 'b').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    if version == "a":
        channels = [[96], [256], [384, 384, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[0], [2], [1, 1, 1]]
        use_lrn = True
    elif version == "b":
        channels = [[64], [192], [384, 256, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[2], [2], [1, 1, 1]]
        use_lrn = False
    else:
        raise ValueError("Unsupported AlexNet version {}".format(version))

    net = alexnet_model(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        use_lrn=use_lrn,
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


def alexnet(**kwargs):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_alexnet(model_name="alexnet", **kwargs)


def alexnetb(**kwargs):
    """
    AlexNet-b model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997. Non-standard version.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_alexnet(version="b", model_name="alexnetb", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        alexnet,
        alexnetb,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != alexnet or weight_count == 62378344)
        assert (model != alexnetb or weight_count == 61100840)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
