"""
    RoR-3 for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.
"""

__all__ = ['CIFARRoR', 'ror3_56_cifar10', 'ror3_56_cifar100', 'ror3_56_svhn', 'ror3_110_cifar10', 'ror3_110_cifar100',
           'ror3_110_svhn', 'ror3_164_cifar10', 'ror3_164_cifar100', 'ror3_164_svhn']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block


class RoRBlock(HybridBlock):
    """
    RoR-3 block for residual path in residual unit.

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
        super(RoRBlock, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class RoRResUnit(HybridBlock):
    """
    RoR-3 residual unit.

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
    last_activate : bool, default True
        Whether activate output.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 dropout_rate,
                 last_activate=True,
                 **kwargs):
        super(RoRResUnit, self).__init__(**kwargs)
        self.last_activate = last_activate
        self.resize_identity = (in_channels != out_channels)

        with self.name_scope():
            self.body = RoRBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                dropout_rate=dropout_rate)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        if self.last_activate:
            x = self.activ(x)
        return x


class RoRResStage(HybridBlock):
    """
    RoR-3 residual stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each unit.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    downsample : bool, default True
        Whether downsample output.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_use_global_stats,
                 dropout_rate,
                 downsample=True,
                 **kwargs):
        super(RoRResStage, self).__init__(**kwargs)
        self.downsample = downsample

        with self.name_scope():
            self.shortcut = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels_list[-1],
                bn_use_global_stats=bn_use_global_stats,
                activation=None)
            self.units = nn.HybridSequential(prefix="")
            with self.units.name_scope():
                for i, out_channels in enumerate(out_channels_list):
                    last_activate = (i != len(out_channels_list) - 1)
                    self.units.add(RoRResUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bn_use_global_stats=bn_use_global_stats,
                        dropout_rate=dropout_rate,
                        last_activate=last_activate))
                    in_channels = out_channels
            if self.downsample:
                self.activ = nn.Activation("relu")
                self.pool = nn.MaxPool2D(
                    pool_size=2,
                    strides=2,
                    padding=0)

    def hybrid_forward(self, F, x):
        identity = self.shortcut(x)
        x = self.units(x)
        x = x + identity
        if self.downsample:
            x = self.activ(x)
            x = self.pool(x)
        return x


class RoRResBody(HybridBlock):
    """
    RoR-3 residual body (main feature path).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_lists : list of list of int
        Number of output channels for each stage.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels_lists,
                 bn_use_global_stats,
                 dropout_rate,
                 **kwargs):
        super(RoRResBody, self).__init__(**kwargs)
        with self.name_scope():
            self.shortcut = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels_lists[-1][-1],
                strides=4,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)
            self.stages = nn.HybridSequential(prefix="")
            with self.stages.name_scope():
                for i, channels_per_stage in enumerate(out_channels_lists):
                    downsample = (i != len(out_channels_lists) - 1)
                    self.stages.add(RoRResStage(
                        in_channels=in_channels,
                        out_channels_list=channels_per_stage,
                        bn_use_global_stats=bn_use_global_stats,
                        dropout_rate=dropout_rate,
                        downsample=downsample))
                    in_channels = channels_per_stage[-1]
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        identity = self.shortcut(x)
        x = self.stages(x)
        x = x + identity
        x = self.activ(x)
        return x


class CIFARRoR(HybridBlock):
    """
    RoR-3 model for CIFAR from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
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
                 bn_use_global_stats=False,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARRoR, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            self.features.add(RoRResBody(
                in_channels=in_channels,
                out_channels_lists=channels,
                bn_use_global_stats=bn_use_global_stats,
                dropout_rate=dropout_rate))
            in_channels = channels[-1][-1]
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


def get_ror_cifar(classes,
                  blocks,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create RoR-3 model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
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

    assert ((blocks - 8) % 6 == 0)
    layers = [(blocks - 8) // 6] * 3

    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFARRoR(
        channels=channels,
        init_block_channels=init_block_channels,
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


def ror3_56_cifar10(classes=10, **kwargs):
    """
    RoR-3-56 model for CIFAR-10 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=56, model_name="ror3_56_cifar10", **kwargs)


def ror3_56_cifar100(classes=100, **kwargs):
    """
    RoR-3-56 model for CIFAR-100 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=56, model_name="ror3_56_cifar100", **kwargs)


def ror3_56_svhn(classes=10, **kwargs):
    """
    RoR-3-56 model for SVHN from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=56, model_name="ror3_56_svhn", **kwargs)


def ror3_110_cifar10(classes=10, **kwargs):
    """
    RoR-3-110 model for CIFAR-10 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=110, model_name="ror3_110_cifar10", **kwargs)


def ror3_110_cifar100(classes=100, **kwargs):
    """
    RoR-3-110 model for CIFAR-100 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=110, model_name="ror3_110_cifar100", **kwargs)


def ror3_110_svhn(classes=10, **kwargs):
    """
    RoR-3-110 model for SVHN from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=110, model_name="ror3_110_svhn", **kwargs)


def ror3_164_cifar10(classes=10, **kwargs):
    """
    RoR-3-164 model for CIFAR-10 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=164, model_name="ror3_164_cifar10", **kwargs)


def ror3_164_cifar100(classes=100, **kwargs):
    """
    RoR-3-164 model for CIFAR-100 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=164, model_name="ror3_164_cifar100", **kwargs)


def ror3_164_svhn(classes=10, **kwargs):
    """
    RoR-3-164 model for SVHN from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

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
    return get_ror_cifar(classes=classes, blocks=164, model_name="ror3_164_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (ror3_56_cifar10, 10),
        (ror3_56_cifar100, 100),
        (ror3_56_svhn, 10),
        (ror3_110_cifar10, 10),
        (ror3_110_cifar100, 100),
        (ror3_110_svhn, 10),
        (ror3_164_cifar10, 10),
        (ror3_164_cifar100, 100),
        (ror3_164_svhn, 10),
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
        assert (model != ror3_56_cifar10 or weight_count == 762746)
        assert (model != ror3_56_cifar100 or weight_count == 768596)
        assert (model != ror3_56_svhn or weight_count == 762746)
        assert (model != ror3_110_cifar10 or weight_count == 1637690)
        assert (model != ror3_110_cifar100 or weight_count == 1643540)
        assert (model != ror3_110_svhn or weight_count == 1637690)
        assert (model != ror3_164_cifar10 or weight_count == 2512634)
        assert (model != ror3_164_cifar100 or weight_count == 2518484)
        assert (model != ror3_164_svhn or weight_count == 2512634)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
