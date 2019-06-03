"""
    X-DenseNet for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.
"""

__all__ = ['CIFARXDenseNet', 'xdensenet40_2_k24_bc_cifar10', 'xdensenet40_2_k24_bc_cifar100',
           'xdensenet40_2_k24_bc_svhn', 'xdensenet40_2_k36_bc_cifar10', 'xdensenet40_2_k36_bc_cifar100',
           'xdensenet40_2_k36_bc_svhn']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, SimpleSequential
from .preresnet import PreResActivation
from .densenet import TransitionBlock
from .xdensenet import pre_xconv3x3_block, XDenseUnit


class XDenseSimpleUnit(Chain):
    """
    X-DenseNet simple unit for CIFAR.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    expand_ratio : int
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate,
                 expand_ratio):
        super(XDenseSimpleUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        inc_channels = out_channels - in_channels

        with self.init_scope():
            self.conv = pre_xconv3x3_block(
                in_channels=in_channels,
                out_channels=inc_channels,
                expand_ratio=expand_ratio)
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


class CIFARXDenseNet(Chain):
    """
    X-DenseNet model for CIFAR from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

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
    expand_ratio : int, default 2
        Ratio of expansion.
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
                 expand_ratio=2,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARXDenseNet, self).__init__()
        self.in_size = in_size
        self.classes = classes
        unit_class = XDenseUnit if bottleneck else XDenseSimpleUnit

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
                                dropout_rate=dropout_rate,
                                expand_ratio=expand_ratio))
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


def get_xdensenet_cifar(classes,
                        blocks,
                        growth_rate,
                        bottleneck,
                        expand_ratio=2,
                        model_name=None,
                        pretrained=False,
                        root=os.path.join("~", ".chainer", "models"),
                        **kwargs):
    """
    Create X-DenseNet model for CIFAR with specific parameters.

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
    expand_ratio : int, default 2
        Ratio of expansion.
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

    net = CIFARXDenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        classes=classes,
        bottleneck=bottleneck,
        expand_ratio=expand_ratio,
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


def xdensenet40_2_k24_bc_cifar10(classes=10, **kwargs):
    """
    X-DenseNet-BC-40-2 (k=24) model for CIFAR-10 from 'Deep Expander Networks: Efficient Deep Networks from Graph
    Theory,' https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet_cifar(classes=classes, blocks=40, growth_rate=24, bottleneck=True,
                               model_name="xdensenet40_2_k24_bc_cifar10", **kwargs)


def xdensenet40_2_k24_bc_cifar100(classes=100, **kwargs):
    """
    X-DenseNet-BC-40-2 (k=24) model for CIFAR-100 from 'Deep Expander Networks: Efficient Deep Networks from Graph
    Theory,' https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet_cifar(classes=classes, blocks=40, growth_rate=24, bottleneck=True,
                               model_name="xdensenet40_2_k24_bc_cifar100", **kwargs)


def xdensenet40_2_k24_bc_svhn(classes=10, **kwargs):
    """
    X-DenseNet-BC-40-2 (k=24) model for SVHN from 'Deep Expander Networks: Efficient Deep Networks from Graph
    Theory,' https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet_cifar(classes=classes, blocks=40, growth_rate=24, bottleneck=True,
                               model_name="xdensenet40_2_k24_bc_svhn", **kwargs)


def xdensenet40_2_k36_bc_cifar10(classes=10, **kwargs):
    """
    X-DenseNet-BC-40-2 (k=36) model for CIFAR-10 from 'Deep Expander Networks: Efficient Deep Networks from Graph
    Theory,' https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet_cifar(classes=classes, blocks=40, growth_rate=36, bottleneck=True,
                               model_name="xdensenet40_2_k36_bc_cifar10", **kwargs)


def xdensenet40_2_k36_bc_cifar100(classes=100, **kwargs):
    """
    X-DenseNet-BC-40-2 (k=36) model for CIFAR-100 from 'Deep Expander Networks: Efficient Deep Networks from Graph
    Theory,' https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet_cifar(classes=classes, blocks=40, growth_rate=36, bottleneck=True,
                               model_name="xdensenet40_2_k36_bc_cifar100", **kwargs)


def xdensenet40_2_k36_bc_svhn(classes=10, **kwargs):
    """
    X-DenseNet-BC-40-2 (k=36) model for SVHN from 'Deep Expander Networks: Efficient Deep Networks from Graph
    Theory,' https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet_cifar(classes=classes, blocks=40, growth_rate=36, bottleneck=True,
                               model_name="xdensenet40_2_k36_bc_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (xdensenet40_2_k24_bc_cifar10, 10),
        (xdensenet40_2_k24_bc_cifar100, 100),
        (xdensenet40_2_k24_bc_svhn, 10),
        (xdensenet40_2_k36_bc_cifar10, 10),
        (xdensenet40_2_k36_bc_cifar100, 100),
        (xdensenet40_2_k36_bc_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xdensenet40_2_k24_bc_cifar10 or weight_count == 690346)
        assert (model != xdensenet40_2_k24_bc_cifar100 or weight_count == 714196)
        assert (model != xdensenet40_2_k24_bc_svhn or weight_count == 690346)
        assert (model != xdensenet40_2_k36_bc_cifar10 or weight_count == 1542682)
        assert (model != xdensenet40_2_k36_bc_cifar100 or weight_count == 1578412)
        assert (model != xdensenet40_2_k36_bc_svhn or weight_count == 1542682)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
