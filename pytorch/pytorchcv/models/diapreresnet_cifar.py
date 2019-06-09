"""
    DIA-PreResNet for CIFAR/SVHN, implemented in PyTorch.
    Original papers: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['CIFARDIAPreResNet', 'diapreresnet20_cifar10', 'diapreresnet20_cifar100', 'diapreresnet20_svhn',
           'diapreresnet56_cifar10', 'diapreresnet56_cifar100', 'diapreresnet56_svhn', 'diapreresnet110_cifar10',
           'diapreresnet110_cifar100', 'diapreresnet110_svhn', 'diapreresnet164bn_cifar10',
           'diapreresnet164bn_cifar100', 'diapreresnet164bn_svhn', 'diapreresnet1001_cifar10',
           'diapreresnet1001_cifar100', 'diapreresnet1001_svhn', 'diapreresnet1202_cifar10',
           'diapreresnet1202_cifar100', 'diapreresnet1202_svhn']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3, DualPathSequential
from .preresnet import PreResActivation
from .diaresnet import DIAAttention
from .diapreresnet import DIAPreResUnit


class CIFARDIAPreResNet(nn.Module):
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
    num_classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARDIAPreResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential(return_two=False)
            attention = DIAAttention(
                in_x_features=channels_per_stage[0],
                in_h_features=channels_per_stage[0])
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), DIAPreResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=False,
                    attention=attention))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("post_activ", PreResActivation(in_channels=in_channels))
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


def get_diapreresnet_cifar(num_classes,
                           blocks,
                           bottleneck,
                           model_name=None,
                           pretrained=False,
                           root=os.path.join("~", ".torch", "models"),
                           **kwargs):
    """
    Create DIA-PreResNet model for CIFAR with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    assert (num_classes in [10, 100])

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


def diapreresnet20_cifar10(num_classes=10, **kwargs):
    """
    DIA-PreResNet-20 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False,
                                  model_name="diapreresnet20_cifar10", **kwargs)


def diapreresnet20_cifar100(num_classes=100, **kwargs):
    """
    DIA-PreResNet-20 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False,
                                  model_name="diapreresnet20_cifar100", **kwargs)


def diapreresnet20_svhn(num_classes=10, **kwargs):
    """
    DIA-PreResNet-20 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False,
                                  model_name="diapreresnet20_svhn", **kwargs)


def diapreresnet56_cifar10(num_classes=10, **kwargs):
    """
    DIA-PreResNet-56 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False,
                                  model_name="diapreresnet56_cifar10", **kwargs)


def diapreresnet56_cifar100(num_classes=100, **kwargs):
    """
    DIA-PreResNet-56 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False,
                                  model_name="diapreresnet56_cifar100", **kwargs)


def diapreresnet56_svhn(num_classes=10, **kwargs):
    """
    DIA-PreResNet-56 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False,
                                  model_name="diapreresnet56_svhn", **kwargs)


def diapreresnet110_cifar10(num_classes=10, **kwargs):
    """
    DIA-PreResNet-110 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False,
                                  model_name="diapreresnet110_cifar10", **kwargs)


def diapreresnet110_cifar100(num_classes=100, **kwargs):
    """
    DIA-PreResNet-110 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False,
                                  model_name="diapreresnet110_cifar100", **kwargs)


def diapreresnet110_svhn(num_classes=10, **kwargs):
    """
    DIA-PreResNet-110 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False,
                                  model_name="diapreresnet110_svhn", **kwargs)


def diapreresnet164bn_cifar10(num_classes=10, **kwargs):
    """
    DIA-PreResNet-164(BN) model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True,
                                  model_name="diapreresnet164bn_cifar10", **kwargs)


def diapreresnet164bn_cifar100(num_classes=100, **kwargs):
    """
    DIA-PreResNet-164(BN) model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True,
                                  model_name="diapreresnet164bn_cifar100", **kwargs)


def diapreresnet164bn_svhn(num_classes=10, **kwargs):
    """
    DIA-PreResNet-164(BN) model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True,
                                  model_name="diapreresnet164bn_svhn", **kwargs)


def diapreresnet1001_cifar10(num_classes=10, **kwargs):
    """
    DIA-PreResNet-1001 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True,
                                  model_name="diapreresnet1001_cifar10", **kwargs)


def diapreresnet1001_cifar100(num_classes=100, **kwargs):
    """
    DIA-PreResNet-1001 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True,
                                  model_name="diapreresnet1001_cifar100", **kwargs)


def diapreresnet1001_svhn(num_classes=10, **kwargs):
    """
    DIA-PreResNet-1001 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True,
                                  model_name="diapreresnet1001_svhn", **kwargs)


def diapreresnet1202_cifar10(num_classes=10, **kwargs):
    """
    DIA-PreResNet-1202 model for CIFAR-10 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False,
                                  model_name="diapreresnet1202_cifar10", **kwargs)


def diapreresnet1202_cifar100(num_classes=100, **kwargs):
    """
    DIA-PreResNet-1202 model for CIFAR-100 from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.
    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False,
                                  model_name="diapreresnet1202_cifar100", **kwargs)


def diapreresnet1202_svhn(num_classes=10, **kwargs):
    """
    DIA-PreResNet-1202 model for SVHN from 'DIANet: Dense-and-Implicit Attention Network,'
    https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False,
                                  model_name="diapreresnet1202_svhn", **kwargs)


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

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
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

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
