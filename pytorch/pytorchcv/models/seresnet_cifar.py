"""
    SE-ResNet for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['CIFARSEResNet', 'seresnet20_cifar10', 'seresnet20_cifar100', 'seresnet20_svhn', 'seresnet56_cifar10',
           'seresnet56_cifar100', 'seresnet56_svhn', 'seresnet110_cifar10', 'seresnet110_cifar100', 'seresnet110_svhn',
           'seresnet164bn_cifar10', 'seresnet164bn_cifar100', 'seresnet164bn_svhn', 'seresnet1001_cifar10',
           'seresnet1001_cifar100', 'seresnet1001_svhn', 'seresnet1202_cifar10', 'seresnet1202_cifar100',
           'seresnet1202_svhn']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3_block
from .seresnet import SEResUnit


class CIFARSEResNet(nn.Module):
    """
    SE-ResNet model for CIFAR from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
        Number of classification num_classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARSEResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SEResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=False))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
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


def get_seresnet_cifar(num_classes,
                       blocks,
                       bottleneck,
                       model_name=None,
                       pretrained=False,
                       root=os.path.join("~", ".torch", "models"),
                       **kwargs):
    """
    Create SE-ResNet model for CIFAR with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification num_classes.
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

    net = CIFARSEResNet(
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


def seresnet20_cifar10(num_classes=10, **kwargs):
    """
    SE-ResNet-20 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False, model_name="seresnet20_cifar10",
                              **kwargs)


def seresnet20_cifar100(num_classes=100, **kwargs):
    """
    SE-ResNet-20 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False, model_name="seresnet20_cifar100",
                              **kwargs)


def seresnet20_svhn(num_classes=10, **kwargs):
    """
    SE-ResNet-20 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False, model_name="seresnet20_svhn",
                              **kwargs)


def seresnet56_cifar10(num_classes=10, **kwargs):
    """
    SE-ResNet-56 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False, model_name="seresnet56_cifar10",
                              **kwargs)


def seresnet56_cifar100(num_classes=100, **kwargs):
    """
    SE-ResNet-56 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False, model_name="seresnet56_cifar100",
                              **kwargs)


def seresnet56_svhn(num_classes=10, **kwargs):
    """
    SE-ResNet-56 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False, model_name="seresnet56_svhn",
                              **kwargs)


def seresnet110_cifar10(num_classes=10, **kwargs):
    """
    SE-ResNet-110 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False, model_name="seresnet110_cifar10",
                              **kwargs)


def seresnet110_cifar100(num_classes=100, **kwargs):
    """
    SE-ResNet-110 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False, model_name="seresnet110_cifar100",
                              **kwargs)


def seresnet110_svhn(num_classes=10, **kwargs):
    """
    SE-ResNet-110 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False, model_name="seresnet110_svhn",
                              **kwargs)


def seresnet164bn_cifar10(num_classes=10, **kwargs):
    """
    SE-ResNet-164(BN) model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True, model_name="seresnet164bn_cifar10",
                              **kwargs)


def seresnet164bn_cifar100(num_classes=100, **kwargs):
    """
    SE-ResNet-164(BN) model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True, model_name="seresnet164bn_cifar100",
                              **kwargs)


def seresnet164bn_svhn(num_classes=10, **kwargs):
    """
    SE-ResNet-164(BN) model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True, model_name="seresnet164bn_svhn",
                              **kwargs)


def seresnet1001_cifar10(num_classes=10, **kwargs):
    """
    SE-ResNet-1001 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True, model_name="seresnet1001_cifar10",
                              **kwargs)


def seresnet1001_cifar100(num_classes=100, **kwargs):
    """
    SE-ResNet-1001 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True, model_name="seresnet1001_cifar100",
                              **kwargs)


def seresnet1001_svhn(num_classes=10, **kwargs):
    """
    SE-ResNet-1001 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True, model_name="seresnet1001_svhn",
                              **kwargs)


def seresnet1202_cifar10(num_classes=10, **kwargs):
    """
    SE-ResNet-1202 model for CIFAR-10 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False, model_name="seresnet1202_cifar10",
                              **kwargs)


def seresnet1202_cifar100(num_classes=100, **kwargs):
    """
    SE-ResNet-1202 model for CIFAR-100 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False,
                              model_name="seresnet1202_cifar100", **kwargs)


def seresnet1202_svhn(num_classes=10, **kwargs):
    """
    SE-ResNet-1202 model for SVHN from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False, model_name="seresnet1202_svhn",
                              **kwargs)


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
        (seresnet20_cifar10, 10),
        (seresnet20_cifar100, 100),
        (seresnet20_svhn, 10),
        (seresnet56_cifar10, 10),
        (seresnet56_cifar100, 100),
        (seresnet56_svhn, 10),
        (seresnet110_cifar10, 10),
        (seresnet110_cifar100, 100),
        (seresnet110_svhn, 10),
        (seresnet164bn_cifar10, 10),
        (seresnet164bn_cifar100, 100),
        (seresnet164bn_svhn, 10),
        (seresnet1001_cifar10, 10),
        (seresnet1001_cifar100, 100),
        (seresnet1001_svhn, 10),
        (seresnet1202_cifar10, 10),
        (seresnet1202_cifar100, 100),
        (seresnet1202_svhn, 10),
    ]

    for model, num_num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != seresnet20_cifar10 or weight_count == 274847)
        assert (model != seresnet20_cifar100 or weight_count == 280697)
        assert (model != seresnet20_svhn or weight_count == 274847)
        assert (model != seresnet56_cifar10 or weight_count == 862889)
        assert (model != seresnet56_cifar100 or weight_count == 868739)
        assert (model != seresnet56_svhn or weight_count == 862889)
        assert (model != seresnet110_cifar10 or weight_count == 1744952)
        assert (model != seresnet110_cifar100 or weight_count == 1750802)
        assert (model != seresnet110_svhn or weight_count == 1744952)
        assert (model != seresnet164bn_cifar10 or weight_count == 1906258)
        assert (model != seresnet164bn_cifar100 or weight_count == 1929388)
        assert (model != seresnet164bn_svhn or weight_count == 1906258)
        assert (model != seresnet1001_cifar10 or weight_count == 11574910)
        assert (model != seresnet1001_cifar100 or weight_count == 11598040)
        assert (model != seresnet1001_svhn or weight_count == 11574910)
        assert (model != seresnet1202_cifar10 or weight_count == 19582226)
        assert (model != seresnet1202_cifar100 or weight_count == 19588076)
        assert (model != seresnet1202_svhn or weight_count == 19582226)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_num_classes))


if __name__ == "__main__":
    _test()
