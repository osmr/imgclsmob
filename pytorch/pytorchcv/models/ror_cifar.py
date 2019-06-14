"""
    RoR-3 for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.
"""

__all__ = ['CIFARRoR', 'ror3_56_cifar10', 'ror3_56_cifar100', 'ror3_56_svhn', 'ror3_110_cifar10', 'ror3_110_cifar100',
           'ror3_110_svhn', 'ror3_164_cifar10', 'ror3_164_cifar100', 'ror3_164_svhn']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block


class RoRBlock(nn.Module):
    """
    RoR-3 block for residual path in residual unit.

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
        super(RoRBlock, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class RoRResUnit(nn.Module):
    """
    RoR-3 residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    last_activate : bool, default True
        Whether activate output.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate,
                 last_activate=True):
        super(RoRResUnit, self).__init__()
        self.last_activate = last_activate
        self.resize_identity = (in_channels != out_channels)

        self.body = RoRBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        if self.last_activate:
            x = self.activ(x)
        return x


class RoRResStage(nn.Module):
    """
    RoR-3 residual stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each unit.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    downsample : bool, default True
        Whether downsample output.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 dropout_rate,
                 downsample=True):
        super(RoRResStage, self).__init__()
        self.downsample = downsample

        self.shortcut = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels_list[-1],
            activation=None)
        self.units = nn.Sequential()
        for i, out_channels in enumerate(out_channels_list):
            last_activate = (i != len(out_channels_list) - 1)
            self.units.add_module("unit{}".format(i + 1), RoRResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate,
                last_activate=last_activate))
            in_channels = out_channels
        if self.downsample:
            self.activ = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.units(x)
        x = x + identity
        if self.downsample:
            x = self.activ(x)
            x = self.pool(x)
        return x


class RoRResBody(nn.Module):
    """
    RoR-3 residual body (main feature path).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_lists : list of list of int
        Number of output channels for each stage.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels_lists,
                 dropout_rate):
        super(RoRResBody, self).__init__()
        self.shortcut = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels_lists[-1][-1],
            stride=4,
            activation=None)
        self.stages = nn.Sequential()
        for i, channels_per_stage in enumerate(out_channels_lists):
            downsample = (i != len(out_channels_lists) - 1)
            self.stages.add_module("stage{}".format(i + 1), RoRResStage(
                in_channels=in_channels,
                out_channels_list=channels_per_stage,
                dropout_rate=dropout_rate,
                downsample=downsample))
            in_channels = channels_per_stage[-1]
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.stages(x)
        x = x + identity
        x = self.activ(x)
        return x


class CIFARRoR(nn.Module):
    """
    RoR-3 model for CIFAR from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    num_classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARRoR, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        self.features.add_module("body", RoRResBody(
            in_channels=in_channels,
            out_channels_lists=channels,
            dropout_rate=dropout_rate))
        in_channels = channels[-1][-1]
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_ror_cifar(num_classes,
                  blocks,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create RoR-3 model for CIFAR with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    assert (num_classes in [10, 100])

    assert ((blocks - 8) % 6 == 0)
    layers = [(blocks - 8) // 6] * 3

    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFARRoR(
        channels=channels,
        init_block_channels=init_block_channels,
        num_classes=num_classes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def ror3_56_cifar10(num_classes=10, **kwargs):
    """
    RoR-3-56 model for CIFAR-10 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=56, model_name="ror3_56_cifar10", **kwargs)


def ror3_56_cifar100(num_classes=100, **kwargs):
    """
    RoR-3-56 model for CIFAR-100 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=56, model_name="ror3_56_cifar100", **kwargs)


def ror3_56_svhn(num_classes=10, **kwargs):
    """
    RoR-3-56 model for SVHN from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=56, model_name="ror3_56_svhn", **kwargs)


def ror3_110_cifar10(num_classes=10, **kwargs):
    """
    RoR-3-110 model for CIFAR-10 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=110, model_name="ror3_110_cifar10", **kwargs)


def ror3_110_cifar100(num_classes=100, **kwargs):
    """
    RoR-3-110 model for CIFAR-100 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=110, model_name="ror3_110_cifar100", **kwargs)


def ror3_110_svhn(num_classes=10, **kwargs):
    """
    RoR-3-110 model for SVHN from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=110, model_name="ror3_110_svhn", **kwargs)


def ror3_164_cifar10(num_classes=10, **kwargs):
    """
    RoR-3-164 model for CIFAR-10 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=164, model_name="ror3_164_cifar10", **kwargs)


def ror3_164_cifar100(num_classes=100, **kwargs):
    """
    RoR-3-164 model for CIFAR-100 from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=164, model_name="ror3_164_cifar100", **kwargs)


def ror3_164_svhn(num_classes=10, **kwargs):
    """
    RoR-3-164 model for SVHN from 'Residual Networks of Residual Networks: Multilevel Residual Networks,'
    https://arxiv.org/abs/1608.02908.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ror_cifar(num_classes=num_classes, blocks=164, model_name="ror3_164_svhn", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

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

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
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

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
