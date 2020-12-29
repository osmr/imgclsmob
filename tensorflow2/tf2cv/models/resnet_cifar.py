"""
    ResNet for CIFAR/SVHN, implemented in TensorFlow.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['CIFARResNet', 'resnet20_cifar10', 'resnet20_cifar100', 'resnet20_svhn',
           'resnet56_cifar10', 'resnet56_cifar100', 'resnet56_svhn',
           'resnet110_cifar10', 'resnet110_cifar100', 'resnet110_svhn',
           'resnet164bn_cifar10', 'resnet164bn_cifar100', 'resnet164bn_svhn',
           'resnet272bn_cifar10', 'resnet272bn_cifar100', 'resnet272bn_svhn',
           'resnet542bn_cifar10', 'resnet542bn_cifar100', 'resnet542bn_svhn',
           'resnet1001_cifar10', 'resnet1001_cifar100', 'resnet1001_svhn',
           'resnet1202_cifar10', 'resnet1202_cifar100', 'resnet1202_svhn']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3_block, SimpleSequential, flatten, is_channels_first
from .resnet import ResUnit


class CIFARResNet(tf.keras.Model):
    """
    ResNet model for CIFAR from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

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
        super(CIFARResNet, self).__init__(**kwargs)
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
                stage.add(ResUnit(
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


def get_resnet_cifar(classes,
                     blocks,
                     bottleneck,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".tensorflow", "models"),
                     **kwargs):
    """
    Create ResNet model for CIFAR with specific parameters.

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

    net = CIFARResNet(
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


def resnet20_cifar10(classes=10, **kwargs):
    """
    ResNet-20 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resnet20_cifar10", **kwargs)


def resnet20_cifar100(classes=100, **kwargs):
    """
    ResNet-20 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resnet20_cifar100", **kwargs)


def resnet20_svhn(classes=10, **kwargs):
    """
    ResNet-20 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resnet20_svhn", **kwargs)


def resnet56_cifar10(classes=10, **kwargs):
    """
    ResNet-56 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="resnet56_cifar10", **kwargs)


def resnet56_cifar100(classes=100, **kwargs):
    """
    ResNet-56 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="resnet56_cifar100", **kwargs)


def resnet56_svhn(classes=10, **kwargs):
    """
    ResNet-56 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="resnet56_svhn", **kwargs)


def resnet110_cifar10(classes=10, **kwargs):
    """
    ResNet-110 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="resnet110_cifar10", **kwargs)


def resnet110_cifar100(classes=100, **kwargs):
    """
    ResNet-110 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="resnet110_cifar100", **kwargs)


def resnet110_svhn(classes=10, **kwargs):
    """
    ResNet-110 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="resnet110_svhn", **kwargs)


def resnet164bn_cifar10(classes=10, **kwargs):
    """
    ResNet-164(BN) model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="resnet164bn_cifar10", **kwargs)


def resnet164bn_cifar100(classes=100, **kwargs):
    """
    ResNet-164(BN) model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="resnet164bn_cifar100", **kwargs)


def resnet164bn_svhn(classes=10, **kwargs):
    """
    ResNet-164(BN) model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="resnet164bn_svhn", **kwargs)


def resnet272bn_cifar10(classes=10, **kwargs):
    """
    ResNet-272(BN) model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="resnet272bn_cifar10", **kwargs)


def resnet272bn_cifar100(classes=100, **kwargs):
    """
    ResNet-272(BN) model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="resnet272bn_cifar100", **kwargs)


def resnet272bn_svhn(classes=10, **kwargs):
    """
    ResNet-272(BN) model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="resnet272bn_svhn", **kwargs)


def resnet542bn_cifar10(classes=10, **kwargs):
    """
    ResNet-542(BN) model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="resnet542bn_cifar10", **kwargs)


def resnet542bn_cifar100(classes=100, **kwargs):
    """
    ResNet-542(BN) model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="resnet542bn_cifar100", **kwargs)


def resnet542bn_svhn(classes=10, **kwargs):
    """
    ResNet-272(BN) model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="resnet542bn_svhn", **kwargs)


def resnet1001_cifar10(classes=10, **kwargs):
    """
    ResNet-1001 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="resnet1001_cifar10", **kwargs)


def resnet1001_cifar100(classes=100, **kwargs):
    """
    ResNet-1001 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="resnet1001_cifar100", **kwargs)


def resnet1001_svhn(classes=10, **kwargs):
    """
    ResNet-1001 model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="resnet1001_svhn", **kwargs)


def resnet1202_cifar10(classes=10, **kwargs):
    """
    ResNet-1202 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="resnet1202_cifar10", **kwargs)


def resnet1202_cifar100(classes=100, **kwargs):
    """
    ResNet-1202 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="resnet1202_cifar100", **kwargs)


def resnet1202_svhn(classes=10, **kwargs):
    """
    ResNet-1202 model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="resnet1202_svhn", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        (resnet20_cifar10, 10),
        (resnet20_cifar100, 100),
        (resnet20_svhn, 10),
        (resnet56_cifar10, 10),
        (resnet56_cifar100, 100),
        (resnet56_svhn, 10),
        (resnet110_cifar10, 10),
        (resnet110_cifar100, 100),
        (resnet110_svhn, 10),
        (resnet164bn_cifar10, 10),
        (resnet164bn_cifar100, 100),
        (resnet164bn_svhn, 10),
        (resnet272bn_cifar10, 10),
        (resnet272bn_cifar100, 100),
        (resnet272bn_svhn, 10),
        (resnet542bn_cifar10, 10),
        (resnet542bn_cifar100, 100),
        (resnet542bn_svhn, 10),
        (resnet1001_cifar10, 10),
        (resnet1001_cifar100, 100),
        (resnet1001_svhn, 10),
        (resnet1202_cifar10, 10),
        (resnet1202_cifar100, 100),
        (resnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 32, 32) if is_channels_first(data_format) else (batch, 32, 32, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnet20_cifar10 or weight_count == 272474)
        assert (model != resnet20_cifar100 or weight_count == 278324)
        assert (model != resnet20_svhn or weight_count == 272474)
        assert (model != resnet56_cifar10 or weight_count == 855770)
        assert (model != resnet56_cifar100 or weight_count == 861620)
        assert (model != resnet56_svhn or weight_count == 855770)
        assert (model != resnet110_cifar10 or weight_count == 1730714)
        assert (model != resnet110_cifar100 or weight_count == 1736564)
        assert (model != resnet110_svhn or weight_count == 1730714)
        assert (model != resnet164bn_cifar10 or weight_count == 1704154)
        assert (model != resnet164bn_cifar100 or weight_count == 1727284)
        assert (model != resnet164bn_svhn or weight_count == 1704154)
        assert (model != resnet272bn_cifar10 or weight_count == 2816986)
        assert (model != resnet272bn_cifar100 or weight_count == 2840116)
        assert (model != resnet272bn_svhn or weight_count == 2816986)
        assert (model != resnet542bn_cifar10 or weight_count == 5599066)
        assert (model != resnet542bn_cifar100 or weight_count == 5622196)
        assert (model != resnet542bn_svhn or weight_count == 5599066)
        assert (model != resnet1001_cifar10 or weight_count == 10328602)
        assert (model != resnet1001_cifar100 or weight_count == 10351732)
        assert (model != resnet1001_svhn or weight_count == 10328602)
        assert (model != resnet1202_cifar10 or weight_count == 19424026)
        assert (model != resnet1202_cifar100 or weight_count == 19429876)
        assert (model != resnet1202_svhn or weight_count == 19424026)


if __name__ == "__main__":
    _test()
