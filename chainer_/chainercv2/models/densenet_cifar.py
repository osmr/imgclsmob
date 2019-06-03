"""
    DenseNet for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

__all__ = ['CIFARDenseNet', 'densenet40_k12_cifar10', 'densenet40_k12_cifar100', 'densenet40_k12_svhn',
           'densenet40_k12_bc_cifar10', 'densenet40_k12_bc_cifar100', 'densenet40_k12_bc_svhn',
           'densenet40_k24_bc_cifar10', 'densenet40_k24_bc_cifar100', 'densenet40_k24_bc_svhn',
           'densenet40_k36_bc_cifar10', 'densenet40_k36_bc_cifar100', 'densenet40_k36_bc_svhn',
           'densenet100_k12_cifar10', 'densenet100_k12_cifar100', 'densenet100_k12_svhn',
           'densenet100_k24_cifar10', 'densenet100_k24_cifar100', 'densenet100_k24_svhn',
           'densenet100_k12_bc_cifar10', 'densenet100_k12_bc_cifar100', 'densenet100_k12_bc_svhn',
           'densenet190_k40_bc_cifar10', 'densenet190_k40_bc_cifar100', 'densenet190_k40_bc_svhn',
           'densenet250_k24_bc_cifar10', 'densenet250_k24_bc_cifar100', 'densenet250_k24_bc_svhn']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, pre_conv3x3_block, SimpleSequential
from .preresnet import PreResActivation
from .densenet import DenseUnit, TransitionBlock


class DenseSimpleUnit(Chain):
    """
    DenseNet simple unit for CIFAR.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate):
        super(DenseSimpleUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        inc_channels = out_channels - in_channels

        with self.init_scope():
            self.conv = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=inc_channels)
            if self.use_dropout:
                self.dropout = partial(
                    F.dropout,
                    ratio=dropout_rate)

    def __call__(self, x):
        identity = x
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.concat((identity, x), axis=1)
        return x


class CIFARDenseNet(Chain):
    """
    DenseNet model for CIFAR from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARDenseNet, self).__init__()
        self.in_size = in_size
        self.classes = classes
        unit_class = DenseUnit if bottleneck else DenseSimpleUnit

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        if i != 0:
                            setattr(stage, "trans{}".format(i + 1), TransitionBlock(
                                in_channels=in_channels,
                                out_channels=(in_channels // 2)))
                            in_channels = in_channels // 2
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), unit_class(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                dropout_rate=dropout_rate))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(in_channels=in_channels))
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


def get_densenet_cifar(classes,
                       blocks,
                       growth_rate,
                       bottleneck,
                       model_name=None,
                       pretrained=False,
                       root=os.path.join("~", ".chainer", "models"),
                       **kwargs):
    """
    Create DenseNet model for CIFAR-10 with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    growth_rate : int
        Growth rate.
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
        assert ((blocks - 4) % 6 == 0)
        layers = [(blocks - 4) // 6] * 3
    else:
        assert ((blocks - 4) % 3 == 0)
        layers = [(blocks - 4) // 3] * 3
    init_block_channels = 2 * growth_rate

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = CIFARDenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        classes=classes,
        bottleneck=bottleneck,
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


def densenet40_k12_cifar10(classes=10, **kwargs):
    """
    DenseNet-40 (k=12) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=12, bottleneck=False,
                              model_name="densenet40_k12_cifar10", **kwargs)


def densenet40_k12_cifar100(classes=100, **kwargs):
    """
    DenseNet-40 (k=12) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=12, bottleneck=False,
                              model_name="densenet40_k12_cifar100", **kwargs)


def densenet40_k12_svhn(classes=10, **kwargs):
    """
    DenseNet-40 (k=12) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=12, bottleneck=False,
                              model_name="densenet40_k12_svhn", **kwargs)


def densenet40_k12_bc_cifar10(classes=10, **kwargs):
    """
    DenseNet-BC-40 (k=12) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=12, bottleneck=True,
                              model_name="densenet40_k12_bc_cifar10", **kwargs)


def densenet40_k12_bc_cifar100(classes=100, **kwargs):
    """
    DenseNet-BC-40 (k=12) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=12, bottleneck=True,
                              model_name="densenet40_k12_bc_cifar100", **kwargs)


def densenet40_k12_bc_svhn(classes=10, **kwargs):
    """
    DenseNet-BC-40 (k=12) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=12, bottleneck=True,
                              model_name="densenet40_k12_bc_svhn", **kwargs)


def densenet40_k24_bc_cifar10(classes=10, **kwargs):
    """
    DenseNet-BC-40 (k=24) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=24, bottleneck=True,
                              model_name="densenet40_k24_bc_cifar10", **kwargs)


def densenet40_k24_bc_cifar100(classes=100, **kwargs):
    """
    DenseNet-BC-40 (k=24) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=24, bottleneck=True,
                              model_name="densenet40_k24_bc_cifar100", **kwargs)


def densenet40_k24_bc_svhn(classes=10, **kwargs):
    """
    DenseNet-BC-40 (k=24) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=24, bottleneck=True,
                              model_name="densenet40_k24_bc_svhn", **kwargs)


def densenet40_k36_bc_cifar10(classes=10, **kwargs):
    """
    DenseNet-BC-40 (k=36) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=36, bottleneck=True,
                              model_name="densenet40_k36_bc_cifar10", **kwargs)


def densenet40_k36_bc_cifar100(classes=100, **kwargs):
    """
    DenseNet-BC-40 (k=36) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=36, bottleneck=True,
                              model_name="densenet40_k36_bc_cifar100", **kwargs)


def densenet40_k36_bc_svhn(classes=10, **kwargs):
    """
    DenseNet-BC-40 (k=36) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=40, growth_rate=36, bottleneck=True,
                              model_name="densenet40_k36_bc_svhn", **kwargs)


def densenet100_k12_cifar10(classes=10, **kwargs):
    """
    DenseNet-100 (k=12) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=12, bottleneck=False,
                              model_name="densenet100_k12_cifar10", **kwargs)


def densenet100_k12_cifar100(classes=100, **kwargs):
    """
    DenseNet-100 (k=12) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=12, bottleneck=False,
                              model_name="densenet100_k12_cifar100", **kwargs)


def densenet100_k12_svhn(classes=10, **kwargs):
    """
    DenseNet-100 (k=12) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=12, bottleneck=False,
                              model_name="densenet100_k12_svhn", **kwargs)


def densenet100_k24_cifar10(classes=10, **kwargs):
    """
    DenseNet-100 (k=24) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=24, bottleneck=False,
                              model_name="densenet100_k24_cifar10", **kwargs)


def densenet100_k24_cifar100(classes=100, **kwargs):
    """
    DenseNet-100 (k=24) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=24, bottleneck=False,
                              model_name="densenet100_k24_cifar100", **kwargs)


def densenet100_k24_svhn(classes=10, **kwargs):
    """
    DenseNet-100 (k=24) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=24, bottleneck=False,
                              model_name="densenet100_k24_svhn", **kwargs)


def densenet100_k12_bc_cifar10(classes=10, **kwargs):
    """
    DenseNet-BC-100 (k=12) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=12, bottleneck=True,
                              model_name="densenet100_k12_bc_cifar10", **kwargs)


def densenet100_k12_bc_cifar100(classes=100, **kwargs):
    """
    DenseNet-BC-100 (k=12) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=12, bottleneck=True,
                              model_name="densenet100_k12_bc_cifar100", **kwargs)


def densenet100_k12_bc_svhn(classes=10, **kwargs):
    """
    DenseNet-BC-100 (k=12) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=100, growth_rate=12, bottleneck=True,
                              model_name="densenet100_k12_bc_svhn", **kwargs)


def densenet190_k40_bc_cifar10(classes=10, **kwargs):
    """
    DenseNet-BC-190 (k=40) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=190, growth_rate=40, bottleneck=True,
                              model_name="densenet190_k40_bc_cifar10", **kwargs)


def densenet190_k40_bc_cifar100(classes=100, **kwargs):
    """
    DenseNet-BC-190 (k=40) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=190, growth_rate=40, bottleneck=True,
                              model_name="densenet190_k40_bc_cifar100", **kwargs)


def densenet190_k40_bc_svhn(classes=10, **kwargs):
    """
    DenseNet-BC-190 (k=40) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=190, growth_rate=40, bottleneck=True,
                              model_name="densenet190_k40_bc_svhn", **kwargs)


def densenet250_k24_bc_cifar10(classes=10, **kwargs):
    """
    DenseNet-BC-250 (k=24) model for CIFAR-10 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=250, growth_rate=24, bottleneck=True,
                              model_name="densenet250_k24_bc_cifar10", **kwargs)


def densenet250_k24_bc_cifar100(classes=100, **kwargs):
    """
    DenseNet-BC-250 (k=24) model for CIFAR-100 from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=250, growth_rate=24, bottleneck=True,
                              model_name="densenet250_k24_bc_cifar100", **kwargs)


def densenet250_k24_bc_svhn(classes=10, **kwargs):
    """
    DenseNet-BC-250 (k=24) model for SVHN from 'Densely Connected Convolutional Networks,'
    https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=250, growth_rate=24, bottleneck=True,
                              model_name="densenet250_k24_bc_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (densenet40_k12_cifar10, 10),
        (densenet40_k12_cifar100, 100),
        (densenet40_k12_svhn, 10),
        (densenet40_k12_bc_cifar10, 10),
        (densenet40_k12_bc_cifar100, 100),
        (densenet40_k12_bc_svhn, 10),
        (densenet40_k24_bc_cifar10, 10),
        (densenet40_k24_bc_cifar100, 100),
        (densenet40_k24_bc_svhn, 10),
        (densenet40_k36_bc_cifar10, 10),
        (densenet40_k36_bc_cifar100, 100),
        (densenet40_k36_bc_svhn, 10),
        (densenet100_k12_cifar10, 10),
        (densenet100_k12_cifar100, 100),
        (densenet100_k12_svhn, 10),
        (densenet100_k24_cifar10, 10),
        (densenet100_k24_cifar100, 100),
        (densenet100_k24_svhn, 10),
        (densenet100_k12_bc_cifar10, 10),
        (densenet100_k12_bc_cifar100, 100),
        (densenet100_k12_bc_svhn, 10),
        (densenet190_k40_bc_cifar10, 10),
        (densenet190_k40_bc_cifar100, 100),
        (densenet190_k40_bc_svhn, 10),
        (densenet250_k24_bc_cifar10, 10),
        (densenet250_k24_bc_cifar100, 100),
        (densenet250_k24_bc_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != densenet40_k12_cifar10 or weight_count == 599050)
        assert (model != densenet40_k12_cifar100 or weight_count == 622360)
        assert (model != densenet40_k12_svhn or weight_count == 599050)
        assert (model != densenet40_k12_bc_cifar10 or weight_count == 176122)
        assert (model != densenet40_k12_bc_cifar100 or weight_count == 188092)
        assert (model != densenet40_k12_bc_svhn or weight_count == 176122)
        assert (model != densenet40_k24_bc_cifar10 or weight_count == 690346)
        assert (model != densenet40_k24_bc_cifar100 or weight_count == 714196)
        assert (model != densenet40_k24_bc_svhn or weight_count == 690346)
        assert (model != densenet40_k36_bc_cifar10 or weight_count == 1542682)
        assert (model != densenet40_k36_bc_cifar100 or weight_count == 1578412)
        assert (model != densenet40_k36_bc_svhn or weight_count == 1542682)
        assert (model != densenet100_k12_cifar10 or weight_count == 4068490)
        assert (model != densenet100_k12_cifar100 or weight_count == 4129600)
        assert (model != densenet100_k12_svhn or weight_count == 4068490)
        assert (model != densenet100_k24_cifar10 or weight_count == 16114138)
        assert (model != densenet100_k24_cifar100 or weight_count == 16236268)
        assert (model != densenet100_k24_svhn or weight_count == 16114138)
        assert (model != densenet100_k12_bc_cifar10 or weight_count == 769162)
        assert (model != densenet100_k12_bc_cifar100 or weight_count == 800032)
        assert (model != densenet100_k12_bc_svhn or weight_count == 769162)
        assert (model != densenet190_k40_bc_cifar10 or weight_count == 25624430)
        assert (model != densenet190_k40_bc_cifar100 or weight_count == 25821620)
        assert (model != densenet190_k40_bc_svhn or weight_count == 25624430)
        assert (model != densenet250_k24_bc_cifar10 or weight_count == 15324406)
        assert (model != densenet250_k24_bc_cifar100 or weight_count == 15480556)
        assert (model != densenet250_k24_bc_svhn or weight_count == 15324406)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
