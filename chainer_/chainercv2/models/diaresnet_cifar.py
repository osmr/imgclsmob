"""
    DIA-ResNet for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['CIFARDIAResNet', 'diaresnet20_cifar10', 'diaresnet20_cifar100', 'diaresnet20_svhn', 'diaresnet56_cifar10',
           'diaresnet56_cifar100', 'diaresnet56_svhn', 'diaresnet110_cifar10', 'diaresnet110_cifar100',
           'diaresnet110_svhn', 'diaresnet164bn_cifar10', 'diaresnet164bn_cifar100', 'diaresnet164bn_svhn',
           'diaresnet1001_cifar10', 'diaresnet1001_cifar100', 'diaresnet1001_svhn', 'diaresnet1202_cifar10',
           'diaresnet1202_cifar100', 'diaresnet1202_svhn']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3_block, DualPathSequential, SimpleSequential
from .diaresnet import DIAAttention, DIAResUnit


class CIFARDIAResNet(Chain):
    """
    DIA-ResNet model for CIFAR from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

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
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARDIAResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = DualPathSequential(return_two=False)
                    attention = DIAAttention(
                        in_x_features=channels_per_stage[0],
                        in_h_features=channels_per_stage[0])
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), DIAResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=False,
                                attention=attention,
                                hold_attention=(j == 0)))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_diaresnet_cifar(classes,
                        blocks,
                        bottleneck,
                        model_name=None,
                        pretrained=False,
                        root=os.path.join("~", ".chainer", "models"),
                        **kwargs):
    """
    Create DIA-ResNet model for CIFAR with specific parameters.

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
    root : str, default '~/.chainer/models'
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

    net = CIFARDIAResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        classes=classes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def diaresnet20_cifar10(classes=10, **kwargs):
    """
    DIA-ResNet-20 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="diaresnet20_cifar10",
                               **kwargs)


def diaresnet20_cifar100(classes=100, **kwargs):
    """
    DIA-ResNet-20 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="diaresnet20_cifar100",
                               **kwargs)


def diaresnet20_svhn(classes=10, **kwargs):
    """
    DIA-ResNet-20 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="diaresnet20_svhn",
                               **kwargs)


def diaresnet56_cifar10(classes=10, **kwargs):
    """
    DIA-ResNet-56 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="diaresnet56_cifar10",
                               **kwargs)


def diaresnet56_cifar100(classes=100, **kwargs):
    """
    DIA-ResNet-56 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="diaresnet56_cifar100",
                               **kwargs)


def diaresnet56_svhn(classes=10, **kwargs):
    """
    DIA-ResNet-56 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="diaresnet56_svhn",
                               **kwargs)


def diaresnet110_cifar10(classes=10, **kwargs):
    """
    DIA-ResNet-110 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="diaresnet110_cifar10",
                               **kwargs)


def diaresnet110_cifar100(classes=100, **kwargs):
    """
    DIA-ResNet-110 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="diaresnet110_cifar100",
                               **kwargs)


def diaresnet110_svhn(classes=10, **kwargs):
    """
    DIA-ResNet-110 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="diaresnet110_svhn",
                               **kwargs)


def diaresnet164bn_cifar10(classes=10, **kwargs):
    """
    DIA-ResNet-164(BN) model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="diaresnet164bn_cifar10",
                               **kwargs)


def diaresnet164bn_cifar100(classes=100, **kwargs):
    """
    DIA-ResNet-164(BN) model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="diaresnet164bn_cifar100",
                               **kwargs)


def diaresnet164bn_svhn(classes=10, **kwargs):
    """
    DIA-ResNet-164(BN) model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="diaresnet164bn_svhn",
                               **kwargs)


def diaresnet1001_cifar10(classes=10, **kwargs):
    """
    DIA-ResNet-1001 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="diaresnet1001_cifar10",
                               **kwargs)


def diaresnet1001_cifar100(classes=100, **kwargs):
    """
    DIA-ResNet-1001 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="diaresnet1001_cifar100",
                               **kwargs)


def diaresnet1001_svhn(classes=10, **kwargs):
    """
    DIA-ResNet-1001 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="diaresnet1001_svhn",
                               **kwargs)


def diaresnet1202_cifar10(classes=10, **kwargs):
    """
    DIA-ResNet-1202 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="diaresnet1202_cifar10",
                               **kwargs)


def diaresnet1202_cifar100(classes=100, **kwargs):
    """
    DIA-ResNet-1202 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="diaresnet1202_cifar100",
                               **kwargs)


def diaresnet1202_svhn(classes=10, **kwargs):
    """
    DIA-ResNet-1202 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="diaresnet1202_svhn",
                               **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (diaresnet20_cifar10, 10),
        (diaresnet20_cifar100, 100),
        (diaresnet20_svhn, 10),
        (diaresnet56_cifar10, 10),
        (diaresnet56_cifar100, 100),
        (diaresnet56_svhn, 10),
        (diaresnet110_cifar10, 10),
        (diaresnet110_cifar100, 100),
        (diaresnet110_svhn, 10),
        (diaresnet164bn_cifar10, 10),
        (diaresnet164bn_cifar100, 100),
        (diaresnet164bn_svhn, 10),
        (diaresnet1001_cifar10, 10),
        (diaresnet1001_cifar100, 100),
        (diaresnet1001_svhn, 10),
        (diaresnet1202_cifar10, 10),
        (diaresnet1202_cifar100, 100),
        (diaresnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diaresnet20_cifar10 or weight_count == 286866)
        assert (model != diaresnet20_cifar100 or weight_count == 292716)
        assert (model != diaresnet20_svhn or weight_count == 286866)
        assert (model != diaresnet56_cifar10 or weight_count == 870162)
        assert (model != diaresnet56_cifar100 or weight_count == 876012)
        assert (model != diaresnet56_svhn or weight_count == 870162)
        assert (model != diaresnet110_cifar10 or weight_count == 1745106)
        assert (model != diaresnet110_cifar100 or weight_count == 1750956)
        assert (model != diaresnet110_svhn or weight_count == 1745106)
        assert (model != diaresnet164bn_cifar10 or weight_count == 1923002)
        assert (model != diaresnet164bn_cifar100 or weight_count == 1946132)
        assert (model != diaresnet164bn_svhn or weight_count == 1923002)
        assert (model != diaresnet1001_cifar10 or weight_count == 10547450)
        assert (model != diaresnet1001_cifar100 or weight_count == 10570580)
        assert (model != diaresnet1001_svhn or weight_count == 10547450)
        assert (model != diaresnet1202_cifar10 or weight_count == 19438418)
        assert (model != diaresnet1202_cifar100 or weight_count == 19444268)
        assert (model != diaresnet1202_svhn or weight_count == 19438418)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
