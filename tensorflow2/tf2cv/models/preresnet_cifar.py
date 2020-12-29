"""
    PreResNet for CIFAR/SVHN, implemented in TensorFlow.
    Original papers: 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
"""

__all__ = ['CIFARPreResNet', 'preresnet20_cifar10', 'preresnet20_cifar100', 'preresnet20_svhn',
           'preresnet56_cifar10', 'preresnet56_cifar100', 'preresnet56_svhn',
           'preresnet110_cifar10', 'preresnet110_cifar100', 'preresnet110_svhn',
           'preresnet164bn_cifar10', 'preresnet164bn_cifar100', 'preresnet164bn_svhn',
           'preresnet272bn_cifar10', 'preresnet272bn_cifar100', 'preresnet272bn_svhn',
           'preresnet542bn_cifar10', 'preresnet542bn_cifar100', 'preresnet542bn_svhn',
           'preresnet1001_cifar10', 'preresnet1001_cifar100', 'preresnet1001_svhn',
           'preresnet1202_cifar10', 'preresnet1202_cifar100', 'preresnet1202_svhn']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3, SimpleSequential, flatten, is_channels_first
from .preresnet import PreResUnit, PreResActivation


class CIFARPreResNet(tf.keras.Model):
    """
    PreResNet model for CIFAR from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

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
        super(CIFARPreResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(PreResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bottleneck=bottleneck,
                    conv1_stride=False,
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


def get_preresnet_cifar(classes,
                        blocks,
                        bottleneck,
                        model_name=None,
                        pretrained=False,
                        root=os.path.join("~", ".tensorflow", "models"),
                        **kwargs):
    """
    Create PreResNet model for CIFAR with specific parameters.

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

    net = CIFARPreResNet(
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


def preresnet20_cifar10(classes=10, **kwargs):
    """
    PreResNet-20 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="preresnet20_cifar10", **kwargs)


def preresnet20_cifar100(classes=100, **kwargs):
    """
    PreResNet-20 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="preresnet20_cifar100",
                               **kwargs)


def preresnet20_svhn(classes=10, **kwargs):
    """
    PreResNet-20 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="preresnet20_svhn", **kwargs)


def preresnet56_cifar10(classes=10, **kwargs):
    """
    PreResNet-56 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="preresnet56_cifar10", **kwargs)


def preresnet56_cifar100(classes=100, **kwargs):
    """
    PreResNet-56 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="preresnet56_cifar100",
                               **kwargs)


def preresnet56_svhn(classes=10, **kwargs):
    """
    PreResNet-56 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="preresnet56_svhn", **kwargs)


def preresnet110_cifar10(classes=10, **kwargs):
    """
    PreResNet-110 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="preresnet110_cifar10",
                               **kwargs)


def preresnet110_cifar100(classes=100, **kwargs):
    """
    PreResNet-110 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="preresnet110_cifar100",
                               **kwargs)


def preresnet110_svhn(classes=10, **kwargs):
    """
    PreResNet-110 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="preresnet110_svhn",
                               **kwargs)


def preresnet164bn_cifar10(classes=10, **kwargs):
    """
    PreResNet-164(BN) model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="preresnet164bn_cifar10",
                               **kwargs)


def preresnet164bn_cifar100(classes=100, **kwargs):
    """
    PreResNet-164(BN) model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="preresnet164bn_cifar100",
                               **kwargs)


def preresnet164bn_svhn(classes=10, **kwargs):
    """
    PreResNet-164(BN) model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="preresnet164bn_svhn",
                               **kwargs)


def preresnet272bn_cifar10(classes=10, **kwargs):
    """
    PreResNet-272(BN) model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="preresnet272bn_cifar10",
                               **kwargs)


def preresnet272bn_cifar100(classes=100, **kwargs):
    """
    PreResNet-272(BN) model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="preresnet272bn_cifar100",
                               **kwargs)


def preresnet272bn_svhn(classes=10, **kwargs):
    """
    PreResNet-272(BN) model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="preresnet272bn_svhn",
                               **kwargs)


def preresnet542bn_cifar10(classes=10, **kwargs):
    """
    PreResNet-542(BN) model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="preresnet542bn_cifar10",
                               **kwargs)


def preresnet542bn_cifar100(classes=100, **kwargs):
    """
    PreResNet-542(BN) model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="preresnet542bn_cifar100",
                               **kwargs)


def preresnet542bn_svhn(classes=10, **kwargs):
    """
    PreResNet-542(BN) model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="preresnet542bn_svhn",
                               **kwargs)


def preresnet1001_cifar10(classes=10, **kwargs):
    """
    PreResNet-1001 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="preresnet1001_cifar10",
                               **kwargs)


def preresnet1001_cifar100(classes=100, **kwargs):
    """
    PreResNet-1001 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="preresnet1001_cifar100",
                               **kwargs)


def preresnet1001_svhn(classes=10, **kwargs):
    """
    PreResNet-1001 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="preresnet1001_svhn",
                               **kwargs)


def preresnet1202_cifar10(classes=10, **kwargs):
    """
    PreResNet-1202 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="preresnet1202_cifar10",
                               **kwargs)


def preresnet1202_cifar100(classes=100, **kwargs):
    """
    PreResNet-1202 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="preresnet1202_cifar100",
                               **kwargs)


def preresnet1202_svhn(classes=10, **kwargs):
    """
    PreResNet-1202 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="preresnet1202_svhn",
                               **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        (preresnet20_cifar10, 10),
        (preresnet20_cifar100, 100),
        (preresnet20_svhn, 10),
        (preresnet56_cifar10, 10),
        (preresnet56_cifar100, 100),
        (preresnet56_svhn, 10),
        (preresnet110_cifar10, 10),
        (preresnet110_cifar100, 100),
        (preresnet110_svhn, 10),
        (preresnet164bn_cifar10, 10),
        (preresnet164bn_cifar100, 100),
        (preresnet164bn_svhn, 10),
        (preresnet272bn_cifar10, 10),
        (preresnet272bn_cifar100, 100),
        (preresnet272bn_svhn, 10),
        (preresnet542bn_cifar10, 10),
        (preresnet542bn_cifar100, 100),
        (preresnet542bn_svhn, 10),
        (preresnet1001_cifar10, 10),
        (preresnet1001_cifar100, 100),
        (preresnet1001_svhn, 10),
        (preresnet1202_cifar10, 10),
        (preresnet1202_cifar100, 100),
        (preresnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 32, 32) if is_channels_first(data_format) else (batch, 32, 32, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != preresnet20_cifar10 or weight_count == 272282)
        assert (model != preresnet20_cifar100 or weight_count == 278132)
        assert (model != preresnet20_svhn or weight_count == 272282)
        assert (model != preresnet56_cifar10 or weight_count == 855578)
        assert (model != preresnet56_cifar100 or weight_count == 861428)
        assert (model != preresnet56_svhn or weight_count == 855578)
        assert (model != preresnet110_cifar10 or weight_count == 1730522)
        assert (model != preresnet110_cifar100 or weight_count == 1736372)
        assert (model != preresnet110_svhn or weight_count == 1730522)
        assert (model != preresnet164bn_cifar10 or weight_count == 1703258)
        assert (model != preresnet164bn_cifar100 or weight_count == 1726388)
        assert (model != preresnet164bn_svhn or weight_count == 1703258)
        assert (model != preresnet272bn_cifar10 or weight_count == 2816090)
        assert (model != preresnet272bn_cifar100 or weight_count == 2839220)
        assert (model != preresnet272bn_svhn or weight_count == 2816090)
        assert (model != preresnet542bn_cifar10 or weight_count == 5598170)
        assert (model != preresnet542bn_cifar100 or weight_count == 5621300)
        assert (model != preresnet542bn_svhn or weight_count == 5598170)
        assert (model != preresnet1001_cifar10 or weight_count == 10327706)
        assert (model != preresnet1001_cifar100 or weight_count == 10350836)
        assert (model != preresnet1001_svhn or weight_count == 10327706)
        assert (model != preresnet1202_cifar10 or weight_count == 19423834)
        assert (model != preresnet1202_cifar100 or weight_count == 19429684)
        assert (model != preresnet1202_svhn or weight_count == 19423834)


if __name__ == "__main__":
    _test()
