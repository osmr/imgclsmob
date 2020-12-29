"""
    SE-PreResNet for CIFAR/SVHN, implemented in TensorFlow.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['CIFARSEPreResNet', 'sepreresnet20_cifar10', 'sepreresnet20_cifar100', 'sepreresnet20_svhn',
           'sepreresnet56_cifar10', 'sepreresnet56_cifar100', 'sepreresnet56_svhn',
           'sepreresnet110_cifar10', 'sepreresnet110_cifar100', 'sepreresnet110_svhn',
           'sepreresnet164bn_cifar10', 'sepreresnet164bn_cifar100', 'sepreresnet164bn_svhn',
           'sepreresnet272bn_cifar10', 'sepreresnet272bn_cifar100', 'sepreresnet272bn_svhn',
           'sepreresnet542bn_cifar10', 'sepreresnet542bn_cifar100', 'sepreresnet542bn_svhn',
           'sepreresnet1001_cifar10', 'sepreresnet1001_cifar100', 'sepreresnet1001_svhn',
           'sepreresnet1202_cifar10', 'sepreresnet1202_cifar100', 'sepreresnet1202_svhn']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3_block, SimpleSequential, flatten, is_channels_first
from .sepreresnet import SEPreResUnit


class CIFARSEPreResNet(tf.keras.Model):
    """
    SE-PreResNet model for CIFAR from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
        super(CIFARSEPreResNet, self).__init__(**kwargs)
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
                stage.add(SEPreResUnit(
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


def get_sepreresnet_cifar(classes,
                          blocks,
                          bottleneck,
                          model_name=None,
                          pretrained=False,
                          root=os.path.join("~", ".tensorflow", "models"),
                          **kwargs):
    """
    Create SE-PreResNet model for CIFAR with specific parameters.

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

    net = CIFARSEPreResNet(
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


def sepreresnet20_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-20 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="sepreresnet20_cifar10",
                                 **kwargs)


def sepreresnet20_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-20 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="sepreresnet20_cifar100",
                                 **kwargs)


def sepreresnet20_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-20 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="sepreresnet20_svhn",
                                 **kwargs)


def sepreresnet56_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-56 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="sepreresnet56_cifar10",
                                 **kwargs)


def sepreresnet56_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-56 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="sepreresnet56_cifar100",
                                 **kwargs)


def sepreresnet56_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-56 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="sepreresnet56_svhn",
                                 **kwargs)


def sepreresnet110_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-110 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="sepreresnet110_cifar10",
                                 **kwargs)


def sepreresnet110_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-110 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="sepreresnet110_cifar100",
                                 **kwargs)


def sepreresnet110_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-110 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="sepreresnet110_svhn",
                                 **kwargs)


def sepreresnet164bn_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-164(BN) model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="sepreresnet164bn_cifar10",
                                 **kwargs)


def sepreresnet164bn_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-164(BN) model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="sepreresnet164bn_cifar100",
                                 **kwargs)


def sepreresnet164bn_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-164(BN) model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="sepreresnet164bn_svhn",
                                 **kwargs)


def sepreresnet272bn_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-272(BN) model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="sepreresnet272bn_cifar10",
                                 **kwargs)


def sepreresnet272bn_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-272(BN) model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="sepreresnet272bn_cifar100",
                                 **kwargs)


def sepreresnet272bn_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-272(BN) model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="sepreresnet272bn_svhn",
                                 **kwargs)


def sepreresnet542bn_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-542(BN) model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="sepreresnet542bn_cifar10",
                                 **kwargs)


def sepreresnet542bn_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-542(BN) model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="sepreresnet542bn_cifar100",
                                 **kwargs)


def sepreresnet542bn_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-542(BN) model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="sepreresnet542bn_svhn",
                                 **kwargs)


def sepreresnet1001_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-1001 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="sepreresnet1001_cifar10",
                                 **kwargs)


def sepreresnet1001_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-1001 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="sepreresnet1001_cifar100",
                                 **kwargs)


def sepreresnet1001_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-1001 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="sepreresnet1001_svhn",
                                 **kwargs)


def sepreresnet1202_cifar10(classes=10, **kwargs):
    """
    SE-PreResNet-1202 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="sepreresnet1202_cifar10",
                                 **kwargs)


def sepreresnet1202_cifar100(classes=100, **kwargs):
    """
    SE-PreResNet-1202 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="sepreresnet1202_cifar100",
                                 **kwargs)


def sepreresnet1202_svhn(classes=10, **kwargs):
    """
    SE-PreResNet-1202 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sepreresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="sepreresnet1202_svhn",
                                 **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        (sepreresnet20_cifar10, 10),
        (sepreresnet20_cifar100, 100),
        (sepreresnet20_svhn, 10),
        (sepreresnet56_cifar10, 10),
        (sepreresnet56_cifar100, 100),
        (sepreresnet56_svhn, 10),
        (sepreresnet110_cifar10, 10),
        (sepreresnet110_cifar100, 100),
        (sepreresnet110_svhn, 10),
        (sepreresnet164bn_cifar10, 10),
        (sepreresnet164bn_cifar100, 100),
        (sepreresnet164bn_svhn, 10),
        (sepreresnet272bn_cifar10, 10),
        (sepreresnet272bn_cifar100, 100),
        (sepreresnet272bn_svhn, 10),
        (sepreresnet542bn_cifar10, 10),
        (sepreresnet542bn_cifar100, 100),
        (sepreresnet542bn_svhn, 10),
        (sepreresnet1001_cifar10, 10),
        (sepreresnet1001_cifar100, 100),
        (sepreresnet1001_svhn, 10),
        (sepreresnet1202_cifar10, 10),
        (sepreresnet1202_cifar100, 100),
        (sepreresnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 32, 32) if is_channels_first(data_format) else (batch, 32, 32, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sepreresnet20_cifar10 or weight_count == 274559)
        assert (model != sepreresnet20_cifar100 or weight_count == 280409)
        assert (model != sepreresnet20_svhn or weight_count == 274559)
        assert (model != sepreresnet56_cifar10 or weight_count == 862601)
        assert (model != sepreresnet56_cifar100 or weight_count == 868451)
        assert (model != sepreresnet56_svhn or weight_count == 862601)
        assert (model != sepreresnet110_cifar10 or weight_count == 1744664)
        assert (model != sepreresnet110_cifar100 or weight_count == 1750514)
        assert (model != sepreresnet110_svhn or weight_count == 1744664)
        assert (model != sepreresnet164bn_cifar10 or weight_count == 1904882)
        assert (model != sepreresnet164bn_cifar100 or weight_count == 1928012)
        assert (model != sepreresnet164bn_svhn or weight_count == 1904882)
        assert (model != sepreresnet272bn_cifar10 or weight_count == 3152450)
        assert (model != sepreresnet272bn_cifar100 or weight_count == 3175580)
        assert (model != sepreresnet272bn_svhn or weight_count == 3152450)
        assert (model != sepreresnet542bn_cifar10 or weight_count == 6271370)
        assert (model != sepreresnet542bn_cifar100 or weight_count == 6294500)
        assert (model != sepreresnet542bn_svhn or weight_count == 6271370)
        assert (model != sepreresnet1001_cifar10 or weight_count == 11573534)
        assert (model != sepreresnet1001_cifar100 or weight_count == 11596664)
        assert (model != sepreresnet1001_svhn or weight_count == 11573534)
        assert (model != sepreresnet1202_cifar10 or weight_count == 19581938)
        assert (model != sepreresnet1202_cifar100 or weight_count == 19587788)
        assert (model != sepreresnet1202_svhn or weight_count == 19581938)


if __name__ == "__main__":
    _test()
