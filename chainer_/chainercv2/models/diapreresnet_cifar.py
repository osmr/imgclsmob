"""
    DIA-PreResNet for CIFAR/SVHN, implemented in Chainer.
    Original papers: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['CIFARDIAPreResNet', 'diapreresnet20_cifar10', 'diapreresnet20_cifar100', 'diapreresnet20_svhn',
           'diapreresnet56_cifar10', 'diapreresnet56_cifar100', 'diapreresnet56_svhn', 'diapreresnet110_cifar10',
           'diapreresnet110_cifar100', 'diapreresnet110_svhn', 'diapreresnet164bn_cifar10',
           'diapreresnet164bn_cifar100', 'diapreresnet164bn_svhn', 'diapreresnet1001_cifar10',
           'diapreresnet1001_cifar100', 'diapreresnet1001_svhn', 'diapreresnet1202_cifar10',
           'diapreresnet1202_cifar100', 'diapreresnet1202_svhn']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, DualPathSequential, SimpleSequential
from .preresnet import PreResActivation
from .diaresnet import DIAAttention
from .diapreresnet import DIAPreResUnit


class CIFARDIAPreResNet(Chain):
    """
    DIA-PreResNet model for CIFAR from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

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
        super(CIFARDIAPreResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3(
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
                            setattr(stage, "unit{}".format(j + 1), DIAPreResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=False,
                                attention=attention,
                                hold_attention=(j == 0)))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(
                    in_channels=in_channels))
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


def get_diapreresnet_cifar(classes,
                           blocks,
                           bottleneck,
                           model_name=None,
                           pretrained=False,
                           root=os.path.join("~", ".chainer", "models"),
                           **kwargs):
    """
    Create DIA-PreResNet model for CIFAR with specific parameters.

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

    net = CIFARDIAPreResNet(
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


def diapreresnet20_cifar10(classes=10, **kwargs):
    """
    DIA-PreResNet-20 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="diapreresnet20_cifar10",
                                  **kwargs)


def diapreresnet20_cifar100(classes=100, **kwargs):
    """
    DIA-PreResNet-20 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="diapreresnet20_cifar100",
                                  **kwargs)


def diapreresnet20_svhn(classes=10, **kwargs):
    """
    DIA-PreResNet-20 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="diapreresnet20_svhn",
                                  **kwargs)


def diapreresnet56_cifar10(classes=10, **kwargs):
    """
    DIA-PreResNet-56 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="diapreresnet56_cifar10",
                                  **kwargs)


def diapreresnet56_cifar100(classes=100, **kwargs):
    """
    DIA-PreResNet-56 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="diapreresnet56_cifar100",
                                  **kwargs)


def diapreresnet56_svhn(classes=10, **kwargs):
    """
    DIA-PreResNet-56 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="diapreresnet56_svhn",
                                  **kwargs)


def diapreresnet110_cifar10(classes=10, **kwargs):
    """
    DIA-PreResNet-110 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="diapreresnet110_cifar10",
                                  **kwargs)


def diapreresnet110_cifar100(classes=100, **kwargs):
    """
    DIA-PreResNet-110 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="diapreresnet110_cifar100",
                                  **kwargs)


def diapreresnet110_svhn(classes=10, **kwargs):
    """
    DIA-PreResNet-110 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="diapreresnet110_svhn",
                                  **kwargs)


def diapreresnet164bn_cifar10(classes=10, **kwargs):
    """
    DIA-PreResNet-164(BN) model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="diapreresnet164bn_cifar10",
                                  **kwargs)


def diapreresnet164bn_cifar100(classes=100, **kwargs):
    """
    DIA-PreResNet-164(BN) model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="diapreresnet164bn_cifar100",
                                  **kwargs)


def diapreresnet164bn_svhn(classes=10, **kwargs):
    """
    DIA-PreResNet-164(BN) model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="diapreresnet164bn_svhn",
                                  **kwargs)


def diapreresnet1001_cifar10(classes=10, **kwargs):
    """
    DIA-PreResNet-1001 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="diapreresnet1001_cifar10",
                                  **kwargs)


def diapreresnet1001_cifar100(classes=100, **kwargs):
    """
    DIA-PreResNet-1001 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="diapreresnet1001_cifar100",
                                  **kwargs)


def diapreresnet1001_svhn(classes=10, **kwargs):
    """
    DIA-PreResNet-1001 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="diapreresnet1001_svhn",
                                  **kwargs)


def diapreresnet1202_cifar10(classes=10, **kwargs):
    """
    DIA-PreResNet-1202 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="diapreresnet1202_cifar10",
                                  **kwargs)


def diapreresnet1202_cifar100(classes=100, **kwargs):
    """
    DIA-PreResNet-1202 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=1202, bottleneck=False,
                                  model_name="diapreresnet1202_cifar100", **kwargs)


def diapreresnet1202_svhn(classes=10, **kwargs):
    """
    DIA-PreResNet-1202 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
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
    return get_diapreresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="diapreresnet1202_svhn",
                                  **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (diapreresnet20_cifar10, 10),
        (diapreresnet20_cifar100, 100),
        (diapreresnet20_svhn, 10),
        (diapreresnet56_cifar10, 10),
        (diapreresnet56_cifar100, 100),
        (diapreresnet56_svhn, 10),
        (diapreresnet110_cifar10, 10),
        (diapreresnet110_cifar100, 100),
        (diapreresnet110_svhn, 10),
        (diapreresnet164bn_cifar10, 10),
        (diapreresnet164bn_cifar100, 100),
        (diapreresnet164bn_svhn, 10),
        (diapreresnet1001_cifar10, 10),
        (diapreresnet1001_cifar100, 100),
        (diapreresnet1001_svhn, 10),
        (diapreresnet1202_cifar10, 10),
        (diapreresnet1202_cifar100, 100),
        (diapreresnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diapreresnet20_cifar10 or weight_count == 286674)
        assert (model != diapreresnet20_cifar100 or weight_count == 292524)
        assert (model != diapreresnet20_svhn or weight_count == 286674)
        assert (model != diapreresnet56_cifar10 or weight_count == 869970)
        assert (model != diapreresnet56_cifar100 or weight_count == 875820)
        assert (model != diapreresnet56_svhn or weight_count == 869970)
        assert (model != diapreresnet110_cifar10 or weight_count == 1744914)
        assert (model != diapreresnet110_cifar100 or weight_count == 1750764)
        assert (model != diapreresnet110_svhn or weight_count == 1744914)
        assert (model != diapreresnet164bn_cifar10 or weight_count == 1922106)
        assert (model != diapreresnet164bn_cifar100 or weight_count == 1945236)
        assert (model != diapreresnet164bn_svhn or weight_count == 1922106)
        assert (model != diapreresnet1001_cifar10 or weight_count == 10546554)
        assert (model != diapreresnet1001_cifar100 or weight_count == 10569684)
        assert (model != diapreresnet1001_svhn or weight_count == 10546554)
        assert (model != diapreresnet1202_cifar10 or weight_count == 19438226)
        assert (model != diapreresnet1202_cifar100 or weight_count == 19444076)
        assert (model != diapreresnet1202_svhn or weight_count == 19438226)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
