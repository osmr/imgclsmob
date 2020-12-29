"""
    SE-ResNet for CIFAR/SVHN, implemented in TensorFlow.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['CIFARSEResNet', 'seresnet20_cifar10', 'seresnet20_cifar100', 'seresnet20_svhn',
           'seresnet56_cifar10', 'seresnet56_cifar100', 'seresnet56_svhn',
           'seresnet110_cifar10', 'seresnet110_cifar100', 'seresnet110_svhn',
           'seresnet164bn_cifar10', 'seresnet164bn_cifar100', 'seresnet164bn_svhn',
           'seresnet272bn_cifar10', 'seresnet272bn_cifar100', 'seresnet272bn_svhn',
           'seresnet542bn_cifar10', 'seresnet542bn_cifar100', 'seresnet542bn_svhn',
           'seresnet1001_cifar10', 'seresnet1001_cifar100', 'seresnet1001_svhn',
           'seresnet1202_cifar10', 'seresnet1202_cifar100', 'seresnet1202_svhn']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3_block, SimpleSequential, flatten, is_channels_first
from .seresnet import SEResUnit


class CIFARSEResNet(tf.keras.Model):
    """
    SE-ResNet model for CIFAR from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
        super(CIFARSEResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(SEResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bottleneck=bottleneck,
                    conv1_stride=False,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
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


def get_seresnet_cifar(classes,
                       blocks,
                       bottleneck,
                       model_name=None,
                       pretrained=False,
                       root=os.path.join("~", ".tensorflow", "models"),
                       **kwargs):
    """
    Create SE-ResNet model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
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

    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = CIFARSEResNet(
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


def seresnet20_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-20 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="seresnet20_cifar10", **kwargs)


def seresnet20_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-20 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="seresnet20_cifar100", **kwargs)


def seresnet20_svhn(classes=10, **kwargs):
    """
    SE-ResNet-20 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="seresnet20_svhn", **kwargs)


def seresnet56_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-56 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="seresnet56_cifar10", **kwargs)


def seresnet56_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-56 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="seresnet56_cifar100", **kwargs)


def seresnet56_svhn(classes=10, **kwargs):
    """
    SE-ResNet-56 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="seresnet56_svhn", **kwargs)


def seresnet110_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-110 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="seresnet110_cifar10", **kwargs)


def seresnet110_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-110 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="seresnet110_cifar100",
                              **kwargs)


def seresnet110_svhn(classes=10, **kwargs):
    """
    SE-ResNet-110 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="seresnet110_svhn", **kwargs)


def seresnet164bn_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-164(BN) model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="seresnet164bn_cifar10",
                              **kwargs)


def seresnet164bn_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-164(BN) model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="seresnet164bn_cifar100",
                              **kwargs)


def seresnet164bn_svhn(classes=10, **kwargs):
    """
    SE-ResNet-164(BN) model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="seresnet164bn_svhn", **kwargs)


def seresnet272bn_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-272(BN) model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="seresnet272bn_cifar10",
                              **kwargs)


def seresnet272bn_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-272(BN) model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="seresnet272bn_cifar100",
                              **kwargs)


def seresnet272bn_svhn(classes=10, **kwargs):
    """
    SE-ResNet-272(BN) model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="seresnet272bn_svhn", **kwargs)


def seresnet542bn_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-542(BN) model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="seresnet542bn_cifar10",
                              **kwargs)


def seresnet542bn_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-542(BN) model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="seresnet542bn_cifar100",
                              **kwargs)


def seresnet542bn_svhn(classes=10, **kwargs):
    """
    SE-ResNet-542(BN) model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="seresnet542bn_svhn", **kwargs)


def seresnet1001_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-1001 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="seresnet1001_cifar10",
                              **kwargs)


def seresnet1001_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-1001 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="seresnet1001_cifar100",
                              **kwargs)


def seresnet1001_svhn(classes=10, **kwargs):
    """
    SE-ResNet-1001 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="seresnet1001_svhn", **kwargs)


def seresnet1202_cifar10(classes=10, **kwargs):
    """
    SE-ResNet-1202 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="seresnet1202_cifar10",
                              **kwargs)


def seresnet1202_cifar100(classes=100, **kwargs):
    """
    SE-ResNet-1202 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="seresnet1202_cifar100",
                              **kwargs)


def seresnet1202_svhn(classes=10, **kwargs):
    """
    SE-ResNet-1202 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="seresnet1202_svhn", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        (seresnet20_cifar10, 10),
        (seresnet20_cifar100, 100),
        (seresnet20_svhn, 10),
        (seresnet56_cifar10, 10),
        (seresnet56_cifar100, 100),
        (seresnet56_svhn, 10),
        (seresnet110_cifar10, 10),
        (seresnet110_cifar100, 100),
        (seresnet110_svhn, 10),
        (seresnet164bn_cifar10, 10),
        (seresnet164bn_cifar100, 100),
        (seresnet164bn_svhn, 10),
        (seresnet272bn_cifar10, 10),
        (seresnet272bn_cifar100, 100),
        (seresnet272bn_svhn, 10),
        (seresnet542bn_cifar10, 10),
        (seresnet542bn_cifar100, 100),
        (seresnet542bn_svhn, 10),
        (seresnet1001_cifar10, 10),
        (seresnet1001_cifar100, 100),
        (seresnet1001_svhn, 10),
        (seresnet1202_cifar10, 10),
        (seresnet1202_cifar100, 100),
        (seresnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 32, 32) if is_channels_first(data_format) else (batch, 32, 32, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != seresnet20_cifar10 or weight_count == 274847)
        assert (model != seresnet20_cifar100 or weight_count == 280697)
        assert (model != seresnet20_svhn or weight_count == 274847)
        assert (model != seresnet56_cifar10 or weight_count == 862889)
        assert (model != seresnet56_cifar100 or weight_count == 868739)
        assert (model != seresnet56_svhn or weight_count == 862889)
        assert (model != seresnet110_cifar10 or weight_count == 1744952)
        assert (model != seresnet110_cifar100 or weight_count == 1750802)
        assert (model != seresnet110_svhn or weight_count == 1744952)
        assert (model != seresnet164bn_cifar10 or weight_count == 1906258)
        assert (model != seresnet164bn_cifar100 or weight_count == 1929388)
        assert (model != seresnet164bn_svhn or weight_count == 1906258)
        assert (model != seresnet272bn_cifar10 or weight_count == 3153826)
        assert (model != seresnet272bn_cifar100 or weight_count == 3176956)
        assert (model != seresnet272bn_svhn or weight_count == 3153826)
        assert (model != seresnet542bn_cifar10 or weight_count == 6272746)
        assert (model != seresnet542bn_cifar100 or weight_count == 6295876)
        assert (model != seresnet542bn_svhn or weight_count == 6272746)
        assert (model != seresnet1001_cifar10 or weight_count == 11574910)
        assert (model != seresnet1001_cifar100 or weight_count == 11598040)
        assert (model != seresnet1001_svhn or weight_count == 11574910)
        assert (model != seresnet1202_cifar10 or weight_count == 19582226)
        assert (model != seresnet1202_cifar100 or weight_count == 19588076)
        assert (model != seresnet1202_svhn or weight_count == 19582226)


if __name__ == "__main__":
    _test()
