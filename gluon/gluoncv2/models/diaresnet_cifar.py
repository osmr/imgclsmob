"""
    DIA-ResNet for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['CIFARDIAResNet', 'diaresnet20_cifar10', 'diaresnet20_cifar100', 'diaresnet20_svhn', 'diaresnet56_cifar10',
           'diaresnet56_cifar100', 'diaresnet56_svhn', 'diaresnet110_cifar10', 'diaresnet110_cifar100',
           'diaresnet110_svhn', 'diaresnet164bn_cifar10', 'diaresnet164bn_cifar100', 'diaresnet164bn_svhn',
           'diaresnet1001_cifar10', 'diaresnet1001_cifar100', 'diaresnet1001_svhn', 'diaresnet1202_cifar10',
           'diaresnet1202_cifar100', 'diaresnet1202_svhn']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3_block, DualPathSequential
from .diaresnet import DIAAttention, DIAResUnit


class CIFARDIAResNet(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARDIAResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(
                    return_two=False,
                    prefix="stage{}_".format(i + 1))
                attention = DIAAttention(
                    in_x_features=channels_per_stage[0],
                    in_h_features=channels_per_stage[0])
                for j, out_channels in enumerate(channels_per_stage):
                    strides = 2 if (j == 0) and (i != 0) else 1
                    with stage.name_scope():
                        stage.add(DIAResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=False,
                            attention=attention))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_diaresnet_cifar(classes,
                        blocks,
                        bottleneck,
                        model_name=None,
                        pretrained=False,
                        ctx=cpu(),
                        root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="diaresnet1202_svhn",
                               **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
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

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
