"""
    DenseNet for CIFAR/SVHN, implemented in Gluon.
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
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3, pre_conv3x3_block
from .preresnet import PreResActivation
from .densenet import DenseUnit, TransitionBlock


class DenseSimpleUnit(HybridBlock):
    """
    DenseNet simple unit for CIFAR.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 dropout_rate,
                 **kwargs):
        super(DenseSimpleUnit, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)
        inc_channels = out_channels - in_channels

        with self.name_scope():
            self.conv = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=inc_channels,
                bn_use_global_stats=bn_use_global_stats)
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        identity = x
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.concat(identity, x, dim=1)
        return x


class CIFARDenseNet(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bn_use_global_stats=False,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARDenseNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        unit_class = DenseUnit if bottleneck else DenseSimpleUnit

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    if i != 0:
                        stage.add(TransitionBlock(
                            in_channels=in_channels,
                            out_channels=(in_channels // 2),
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = in_channels // 2
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(unit_class(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bn_use_global_stats=bn_use_global_stats,
                            dropout_rate=dropout_rate))
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


def get_densenet_cifar(classes,
                       blocks,
                       growth_rate,
                       bottleneck,
                       model_name=None,
                       pretrained=False,
                       ctx=cpu(),
                       root=os.path.join("~", ".mxnet", "models"),
                       **kwargs):
    """
    Create DenseNet model for CIFAR with specific parameters.

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar(classes=classes, blocks=250, growth_rate=24, bottleneck=True,
                              model_name="densenet250_k24_bc_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
