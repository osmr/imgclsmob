"""
    DIA-PreResNet for CIFAR/SVHN, implemented in Gluon.
    Original papers: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['CIFARDIAPreResNet', 'diapreresnet20_cifar10', 'diapreresnet20_cifar100', 'diapreresnet20_svhn',
           'diapreresnet56_cifar10', 'diapreresnet56_cifar100', 'diapreresnet56_svhn', 'diapreresnet110_cifar10',
           'diapreresnet110_cifar100', 'diapreresnet110_svhn', 'diapreresnet164bn_cifar10',
           'diapreresnet164bn_cifar100', 'diapreresnet164bn_svhn', 'diapreresnet1001_cifar10',
           'diapreresnet1001_cifar100', 'diapreresnet1001_svhn', 'diapreresnet1202_cifar10',
           'diapreresnet1202_cifar100', 'diapreresnet1202_svhn']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3, DualPathSequential
from .preresnet import PreResActivation
from .diaresnet import DIAAttention
from .diapreresnet import DIAPreResUnit


class CIFARDIAPreResNet(HybridBlock):
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
                 classes=10,
                 **kwargs):
        super(CIFARDIAPreResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(
                    return_two=False,
                    prefix="stage{}_".format(i + 1))
                attention = DIAAttention(
                    in_x_features=channels_per_stage[0],
                    in_h_features=channels_per_stage[0])
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(DIAPreResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=False,
                            attention=attention))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PreResActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
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


def get_diapreresnet_cifar(classes,
                           blocks,
                           bottleneck,
                           model_name=None,
                           pretrained=False,
                           ctx=cpu(),
                           root=os.path.join("~", ".mxnet", "models"),
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="diapreresnet1202_svhn",
                                  **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
