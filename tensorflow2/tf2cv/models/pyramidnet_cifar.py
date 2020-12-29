"""
    PyramidNet for CIFAR/SVHN, implemented in TensorFlow.
    Original paper: 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.
"""

__all__ = ['CIFARPyramidNet', 'pyramidnet110_a48_cifar10', 'pyramidnet110_a48_cifar100', 'pyramidnet110_a48_svhn',
           'pyramidnet110_a84_cifar10', 'pyramidnet110_a84_cifar100', 'pyramidnet110_a84_svhn',
           'pyramidnet110_a270_cifar10', 'pyramidnet110_a270_cifar100', 'pyramidnet110_a270_svhn',
           'pyramidnet164_a270_bn_cifar10', 'pyramidnet164_a270_bn_cifar100', 'pyramidnet164_a270_bn_svhn',
           'pyramidnet200_a240_bn_cifar10', 'pyramidnet200_a240_bn_cifar100', 'pyramidnet200_a240_bn_svhn',
           'pyramidnet236_a220_bn_cifar10', 'pyramidnet236_a220_bn_cifar100', 'pyramidnet236_a220_bn_svhn',
           'pyramidnet272_a200_bn_cifar10', 'pyramidnet272_a200_bn_cifar100', 'pyramidnet272_a200_bn_svhn']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3_block, SimpleSequential, flatten, is_channels_first
from .preresnet import PreResActivation
from .pyramidnet import PyrUnit


class CIFARPyramidNet(tf.keras.Model):
    """
    PyramidNet model for CIFAR from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 data_format="channels_last",
                 **kwargs):
        super(CIFARPyramidNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            activation=None,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(PyrUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bottleneck=bottleneck,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(PreResActivation(
            in_channels=in_channels,
            data_format=data_format,
            name="post_activ"))
        self.features.add(nn.AveragePooling2D(
            pool_size=8,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_pyramidnet_cifar(classes,
                         blocks,
                         alpha,
                         bottleneck,
                         model_name=None,
                         pretrained=False,
                         root=os.path.join("~", ".tensorflow", "models"),
                         **kwargs):
    """
    Create PyramidNet for CIFAR model with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    alpha : int
        PyramidNet's alpha value.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    assert (classes in [10, 100])

    if bottleneck:
        assert ((blocks - 2) % 9 == 0)
        layers = [(blocks - 2) // 9] * 3
    else:
        assert ((blocks - 2) % 6 == 0)
        layers = [(blocks - 2) // 6] * 3
    init_block_channels = 16

    growth_add = float(alpha) / float(sum(layers))
    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [[(i + 1) * growth_add + xi[-1][-1] for i in list(range(yi))]],
        layers,
        [[init_block_channels]])[1:]
    channels = [[int(round(cij)) for cij in ci] for ci in channels]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = CIFARPyramidNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        classes=classes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def pyramidnet110_a48_cifar10(classes=10, **kwargs):
    """
    PyramidNet-110 (a=48) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=48,
        bottleneck=False,
        model_name="pyramidnet110_a48_cifar10",
        **kwargs)


def pyramidnet110_a48_cifar100(classes=100, **kwargs):
    """
    PyramidNet-110 (a=48) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=48,
        bottleneck=False,
        model_name="pyramidnet110_a48_cifar100",
        **kwargs)


def pyramidnet110_a48_svhn(classes=10, **kwargs):
    """
    PyramidNet-110 (a=48) model for SVHN from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=48,
        bottleneck=False,
        model_name="pyramidnet110_a48_svhn",
        **kwargs)


def pyramidnet110_a84_cifar10(classes=10, **kwargs):
    """
    PyramidNet-110 (a=84) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=84,
        bottleneck=False,
        model_name="pyramidnet110_a84_cifar10",
        **kwargs)


def pyramidnet110_a84_cifar100(classes=100, **kwargs):
    """
    PyramidNet-110 (a=84) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=84,
        bottleneck=False,
        model_name="pyramidnet110_a84_cifar100",
        **kwargs)


def pyramidnet110_a84_svhn(classes=10, **kwargs):
    """
    PyramidNet-110 (a=84) model for SVHN from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=84,
        bottleneck=False,
        model_name="pyramidnet110_a84_svhn",
        **kwargs)


def pyramidnet110_a270_cifar10(classes=10, **kwargs):
    """
    PyramidNet-110 (a=270) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=270,
        bottleneck=False,
        model_name="pyramidnet110_a270_cifar10",
        **kwargs)


def pyramidnet110_a270_cifar100(classes=100, **kwargs):
    """
    PyramidNet-110 (a=270) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=270,
        bottleneck=False,
        model_name="pyramidnet110_a270_cifar100",
        **kwargs)


def pyramidnet110_a270_svhn(classes=10, **kwargs):
    """
    PyramidNet-110 (a=270) model for SVHN from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=110,
        alpha=270,
        bottleneck=False,
        model_name="pyramidnet110_a270_svhn",
        **kwargs)


def pyramidnet164_a270_bn_cifar10(classes=10, **kwargs):
    """
    PyramidNet-164 (a=270, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=164,
        alpha=270,
        bottleneck=True,
        model_name="pyramidnet164_a270_bn_cifar10",
        **kwargs)


def pyramidnet164_a270_bn_cifar100(classes=100, **kwargs):
    """
    PyramidNet-164 (a=270, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=164,
        alpha=270,
        bottleneck=True,
        model_name="pyramidnet164_a270_bn_cifar100",
        **kwargs)


def pyramidnet164_a270_bn_svhn(classes=10, **kwargs):
    """
    PyramidNet-164 (a=270, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=164,
        alpha=270,
        bottleneck=True,
        model_name="pyramidnet164_a270_bn_svhn",
        **kwargs)


def pyramidnet200_a240_bn_cifar10(classes=10, **kwargs):
    """
    PyramidNet-200 (a=240, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=200,
        alpha=240,
        bottleneck=True,
        model_name="pyramidnet200_a240_bn_cifar10",
        **kwargs)


def pyramidnet200_a240_bn_cifar100(classes=100, **kwargs):
    """
    PyramidNet-200 (a=240, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=200,
        alpha=240,
        bottleneck=True,
        model_name="pyramidnet200_a240_bn_cifar100",
        **kwargs)


def pyramidnet200_a240_bn_svhn(classes=10, **kwargs):
    """
    PyramidNet-200 (a=240, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=200,
        alpha=240,
        bottleneck=True,
        model_name="pyramidnet200_a240_bn_svhn",
        **kwargs)


def pyramidnet236_a220_bn_cifar10(classes=10, **kwargs):
    """
    PyramidNet-236 (a=220, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=236,
        alpha=220,
        bottleneck=True,
        model_name="pyramidnet236_a220_bn_cifar10",
        **kwargs)


def pyramidnet236_a220_bn_cifar100(classes=100, **kwargs):
    """
    PyramidNet-236 (a=220, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=236,
        alpha=220,
        bottleneck=True,
        model_name="pyramidnet236_a220_bn_cifar100",
        **kwargs)


def pyramidnet236_a220_bn_svhn(classes=10, **kwargs):
    """
    PyramidNet-236 (a=220, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=236,
        alpha=220,
        bottleneck=True,
        model_name="pyramidnet236_a220_bn_svhn",
        **kwargs)


def pyramidnet272_a200_bn_cifar10(classes=10, **kwargs):
    """
    PyramidNet-272 (a=200, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=272,
        alpha=200,
        bottleneck=True,
        model_name="pyramidnet272_a200_bn_cifar10",
        **kwargs)


def pyramidnet272_a200_bn_cifar100(classes=100, **kwargs):
    """
    PyramidNet-272 (a=200, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=272,
        alpha=200,
        bottleneck=True,
        model_name="pyramidnet272_a200_bn_cifar100",
        **kwargs)


def pyramidnet272_a200_bn_svhn(classes=10, **kwargs):
    """
    PyramidNet-272 (a=200, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        classes=classes,
        blocks=272,
        alpha=200,
        bottleneck=True,
        model_name="pyramidnet272_a200_bn_svhn",
        **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        (pyramidnet110_a48_cifar10, 10),
        (pyramidnet110_a48_cifar100, 100),
        (pyramidnet110_a48_svhn, 10),
        (pyramidnet110_a84_cifar10, 10),
        (pyramidnet110_a84_cifar100, 100),
        (pyramidnet110_a84_svhn, 10),
        (pyramidnet110_a270_cifar10, 10),
        (pyramidnet110_a270_cifar100, 100),
        (pyramidnet110_a270_svhn, 10),
        (pyramidnet164_a270_bn_cifar10, 10),
        (pyramidnet164_a270_bn_cifar100, 100),
        (pyramidnet164_a270_bn_svhn, 10),
        (pyramidnet200_a240_bn_cifar10, 10),
        (pyramidnet200_a240_bn_cifar100, 100),
        (pyramidnet200_a240_bn_svhn, 10),
        (pyramidnet236_a220_bn_cifar10, 10),
        (pyramidnet236_a220_bn_cifar100, 100),
        (pyramidnet236_a220_bn_svhn, 10),
        (pyramidnet272_a200_bn_cifar10, 10),
        (pyramidnet272_a200_bn_cifar100, 100),
        (pyramidnet272_a200_bn_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 32, 32) if is_channels_first(data_format) else (batch, 32, 32, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pyramidnet110_a48_cifar10 or weight_count == 1772706)
        assert (model != pyramidnet110_a48_cifar100 or weight_count == 1778556)
        assert (model != pyramidnet110_a48_svhn or weight_count == 1772706)
        assert (model != pyramidnet110_a84_cifar10 or weight_count == 3904446)
        assert (model != pyramidnet110_a84_cifar100 or weight_count == 3913536)
        assert (model != pyramidnet110_a84_svhn or weight_count == 3904446)
        assert (model != pyramidnet110_a270_cifar10 or weight_count == 28485477)
        assert (model != pyramidnet110_a270_cifar100 or weight_count == 28511307)
        assert (model != pyramidnet110_a270_svhn or weight_count == 28485477)
        assert (model != pyramidnet164_a270_bn_cifar10 or weight_count == 27216021)
        assert (model != pyramidnet164_a270_bn_cifar100 or weight_count == 27319071)
        assert (model != pyramidnet164_a270_bn_svhn or weight_count == 27216021)
        assert (model != pyramidnet200_a240_bn_cifar10 or weight_count == 26752702)
        assert (model != pyramidnet200_a240_bn_cifar100 or weight_count == 26844952)
        assert (model != pyramidnet200_a240_bn_svhn or weight_count == 26752702)
        assert (model != pyramidnet236_a220_bn_cifar10 or weight_count == 26969046)
        assert (model != pyramidnet236_a220_bn_cifar100 or weight_count == 27054096)
        assert (model != pyramidnet236_a220_bn_svhn or weight_count == 26969046)
        assert (model != pyramidnet272_a200_bn_cifar10 or weight_count == 26210842)
        assert (model != pyramidnet272_a200_bn_cifar100 or weight_count == 26288692)
        assert (model != pyramidnet272_a200_bn_svhn or weight_count == 26210842)


if __name__ == "__main__":
    _test()
