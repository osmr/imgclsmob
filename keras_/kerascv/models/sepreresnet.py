"""
    SE-PreResNet, implemented in Keras.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['sepreresnet', 'sepreresnet18', 'sepreresnet34', 'sepreresnet50', 'sepreresnet50b', 'sepreresnet101',
           'sepreresnet101b', 'sepreresnet152', 'sepreresnet152b', 'sepreresnet200', 'sepreresnet200b']

import os
from keras import layers as nn
from keras.models import Model
from .common import conv1x1, se_block, is_channels_first, flatten
from .preresnet import preres_block, preres_bottleneck_block, preres_init_block, preres_activation


def sepreres_unit(x,
                  in_channels,
                  out_channels,
                  strides,
                  bottleneck,
                  conv1_stride,
                  name="sepreres_unit"):
    """
    SE-PreResNet unit.

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
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    name : str, default 'sepreres_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor.
    """
    identity = x

    if bottleneck:
        x, x_pre_activ = preres_bottleneck_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            conv1_stride=conv1_stride,
            name=name + "/body")
    else:
        x, x_pre_activ = preres_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            name=name + "/body")

    x = se_block(
        x=x,
        channels=out_channels,
        name=name + "/se")

    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        identity = conv1x1(
            x=x_pre_activ,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            name=name + "/identity_conv")

    x = nn.add([x, identity], name=name + "/add")
    return x


def sepreresnet(channels,
                init_block_channels,
                bottleneck,
                conv1_stride,
                in_channels=3,
                in_size=(224, 224),
                classes=1000):
    """
    SE-PreResNet model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    input_shape = (in_channels, 224, 224) if is_channels_first() else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = preres_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            x = sepreres_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bottleneck=bottleneck,
                conv1_stride=conv1_stride,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = preres_activation(
        x=x,
        name="features/post_activ")
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


def get_sepreresnet(blocks,
                    conv1_stride=True,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join('~', '.keras', 'models'),
                    **kwargs):
    """
    Create PreResNet or SE-PreResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported SE-PreResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = sepreresnet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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


def sepreresnet18(**kwargs):
    """
    SE-PreResNet-18 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=18, model_name="sepreresnet18", **kwargs)


def sepreresnet34(**kwargs):
    """
    SE-PreResNet-34 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=34, model_name="sepreresnet34", **kwargs)


def sepreresnet50(**kwargs):
    """
    SE-PreResNet-50 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=50, model_name="sepreresnet50", **kwargs)


def sepreresnet50b(**kwargs):
    """
    SE-PreResNet-50 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=50, conv1_stride=False, model_name="sepreresnet50b", **kwargs)


def sepreresnet101(**kwargs):
    """
    SE-PreResNet-101 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=101, model_name="sepreresnet101", **kwargs)


def sepreresnet101b(**kwargs):
    """
    SE-PreResNet-101 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=101, conv1_stride=False, model_name="sepreresnet101b", **kwargs)


def sepreresnet152(**kwargs):
    """
    SE-PreResNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=152, model_name="sepreresnet152", **kwargs)


def sepreresnet152b(**kwargs):
    """
    SE-PreResNet-152 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=152, conv1_stride=False, model_name="sepreresnet152b", **kwargs)


def sepreresnet200(**kwargs):
    """
    SE-PreResNet-200 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507. It's an
    experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=200, model_name="sepreresnet200", **kwargs)


def sepreresnet200b(**kwargs):
    """
    SE-PreResNet-200 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet(blocks=200, conv1_stride=False, model_name="sepreresnet200b", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        sepreresnet18,
        sepreresnet34,
        sepreresnet50,
        sepreresnet50b,
        sepreresnet101,
        sepreresnet101b,
        sepreresnet152,
        sepreresnet152b,
        sepreresnet200,
        sepreresnet200b,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sepreresnet18 or weight_count == 11776928)
        assert (model != sepreresnet34 or weight_count == 21957204)
        assert (model != sepreresnet50 or weight_count == 28080472)
        assert (model != sepreresnet50b or weight_count == 28080472)
        assert (model != sepreresnet101 or weight_count == 49319320)
        assert (model != sepreresnet101b or weight_count == 49319320)
        assert (model != sepreresnet152 or weight_count == 66814296)
        assert (model != sepreresnet152b or weight_count == 66814296)
        assert (model != sepreresnet200 or weight_count == 71828312)
        assert (model != sepreresnet200b or weight_count == 71828312)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
