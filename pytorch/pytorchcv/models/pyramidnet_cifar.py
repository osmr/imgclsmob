"""
    PyramidNet for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.
"""

__all__ = ['CIFARPyramidNet', 'pyramidnet110_a48_cifar10', 'pyramidnet110_a48_cifar100', 'pyramidnet110_a48_svhn',
           'pyramidnet110_a84_cifar10', 'pyramidnet110_a84_cifar100', 'pyramidnet110_a84_svhn',
           'pyramidnet110_a270_cifar10', 'pyramidnet110_a270_cifar100', 'pyramidnet110_a270_svhn',
           'pyramidnet164_a270_bn_cifar10', 'pyramidnet164_a270_bn_cifar100', 'pyramidnet164_a270_bn_svhn',
           'pyramidnet200_a240_bn_cifar10', 'pyramidnet200_a240_bn_cifar100', 'pyramidnet200_a240_bn_svhn',
           'pyramidnet236_a220_bn_cifar10', 'pyramidnet236_a220_bn_cifar100', 'pyramidnet236_a220_bn_svhn',
           'pyramidnet272_a200_bn_cifar10', 'pyramidnet272_a200_bn_cifar100', 'pyramidnet272_a200_bn_svhn']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3_block
from .preresnet import PreResActivation
from .pyramidnet import PyrUnit


class CIFARPyramidNet(nn.Module):
    """
    PyramidNet model for CIFAR from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

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
        super(CIFARPyramidNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            activation=None))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 1 if (i == 0) or (j != 0) else 2
                stage.add_module("unit{}".format(j + 1), PyrUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('post_activ', PreResActivation(in_channels=in_channels))
        self.features.add_module('final_pool', nn.AvgPool2d(
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


def get_pyramidnet_cifar(num_classes,
                         blocks,
                         alpha,
                         bottleneck,
                         model_name=None,
                         pretrained=False,
                         root=os.path.join("~", ".torch", "models"),
                         **kwargs):
    """
    Create PyramidNet for CIFAR model with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    alpha : int
        PyramidNet's alpha value.
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
    init_block_channels = 16

    growth_add = float(alpha) / float(sum(layers))
    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [[(i + 1) * growth_add + xi[-1][-1] for i in list(range(yi))]],
        layers,
        [[init_block_channels]])[1:]
    channels = [[int(round(cij)) for cij in ci] for ci in channels]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = CIFARPyramidNet(
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


def pyramidnet110_a48_cifar10(num_classes=10, **kwargs):
    """
    PyramidNet-110 (a=48) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=48,
        bottleneck=False,
        model_name="pyramidnet110_a48_cifar10",
        **kwargs)


def pyramidnet110_a48_cifar100(num_classes=100, **kwargs):
    """
    PyramidNet-110 (a=48) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=48,
        bottleneck=False,
        model_name="pyramidnet110_a48_cifar100",
        **kwargs)


def pyramidnet110_a48_svhn(num_classes=10, **kwargs):
    """
    PyramidNet-110 (a=48) model for SVHN from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=48,
        bottleneck=False,
        model_name="pyramidnet110_a48_svhn",
        **kwargs)


def pyramidnet110_a84_cifar10(num_classes=10, **kwargs):
    """
    PyramidNet-110 (a=84) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=84,
        bottleneck=False,
        model_name="pyramidnet110_a84_cifar10",
        **kwargs)


def pyramidnet110_a84_cifar100(num_classes=100, **kwargs):
    """
    PyramidNet-110 (a=84) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=84,
        bottleneck=False,
        model_name="pyramidnet110_a84_cifar100",
        **kwargs)


def pyramidnet110_a84_svhn(num_classes=10, **kwargs):
    """
    PyramidNet-110 (a=84) model for SVHN from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=84,
        bottleneck=False,
        model_name="pyramidnet110_a84_svhn",
        **kwargs)


def pyramidnet110_a270_cifar10(num_classes=10, **kwargs):
    """
    PyramidNet-110 (a=270) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=270,
        bottleneck=False,
        model_name="pyramidnet110_a270_cifar10",
        **kwargs)


def pyramidnet110_a270_cifar100(num_classes=100, **kwargs):
    """
    PyramidNet-110 (a=270) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=270,
        bottleneck=False,
        model_name="pyramidnet110_a270_cifar100",
        **kwargs)


def pyramidnet110_a270_svhn(num_classes=10, **kwargs):
    """
    PyramidNet-110 (a=270) model for SVHN from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=110,
        alpha=270,
        bottleneck=False,
        model_name="pyramidnet110_a270_svhn",
        **kwargs)


def pyramidnet164_a270_bn_cifar10(num_classes=10, **kwargs):
    """
    PyramidNet-164 (a=270, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=164,
        alpha=270,
        bottleneck=True,
        model_name="pyramidnet164_a270_bn_cifar10",
        **kwargs)


def pyramidnet164_a270_bn_cifar100(num_classes=100, **kwargs):
    """
    PyramidNet-164 (a=270, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=164,
        alpha=270,
        bottleneck=True,
        model_name="pyramidnet164_a270_bn_cifar100",
        **kwargs)


def pyramidnet164_a270_bn_svhn(num_classes=10, **kwargs):
    """
    PyramidNet-164 (a=270, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=164,
        alpha=270,
        bottleneck=True,
        model_name="pyramidnet164_a270_bn_svhn",
        **kwargs)


def pyramidnet200_a240_bn_cifar10(num_classes=10, **kwargs):
    """
    PyramidNet-200 (a=240, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=200,
        alpha=240,
        bottleneck=True,
        model_name="pyramidnet200_a240_bn_cifar10",
        **kwargs)


def pyramidnet200_a240_bn_cifar100(num_classes=100, **kwargs):
    """
    PyramidNet-200 (a=240, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=200,
        alpha=240,
        bottleneck=True,
        model_name="pyramidnet200_a240_bn_cifar100",
        **kwargs)


def pyramidnet200_a240_bn_svhn(num_classes=10, **kwargs):
    """
    PyramidNet-200 (a=240, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=200,
        alpha=240,
        bottleneck=True,
        model_name="pyramidnet200_a240_bn_svhn",
        **kwargs)


def pyramidnet236_a220_bn_cifar10(num_classes=10, **kwargs):
    """
    PyramidNet-236 (a=220, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=236,
        alpha=220,
        bottleneck=True,
        model_name="pyramidnet236_a220_bn_cifar10",
        **kwargs)


def pyramidnet236_a220_bn_cifar100(num_classes=100, **kwargs):
    """
    PyramidNet-236 (a=220, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=236,
        alpha=220,
        bottleneck=True,
        model_name="pyramidnet236_a220_bn_cifar100",
        **kwargs)


def pyramidnet236_a220_bn_svhn(num_classes=10, **kwargs):
    """
    PyramidNet-236 (a=220, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=236,
        alpha=220,
        bottleneck=True,
        model_name="pyramidnet236_a220_bn_svhn",
        **kwargs)


def pyramidnet272_a200_bn_cifar10(num_classes=10, **kwargs):
    """
    PyramidNet-272 (a=200, bn) model for CIFAR-10 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=272,
        alpha=200,
        bottleneck=True,
        model_name="pyramidnet272_a200_bn_cifar10",
        **kwargs)


def pyramidnet272_a200_bn_cifar100(num_classes=100, **kwargs):
    """
    PyramidNet-272 (a=200, bn) model for CIFAR-100 from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=272,
        alpha=200,
        bottleneck=True,
        model_name="pyramidnet272_a200_bn_cifar100",
        **kwargs)


def pyramidnet272_a200_bn_svhn(num_classes=10, **kwargs):
    """
    PyramidNet-272 (a=200, bn) model for SVHN from 'Deep Pyramidal Residual Networks,'
    https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet_cifar(
        num_classes=num_classes,
        blocks=272,
        alpha=200,
        bottleneck=True,
        model_name="pyramidnet272_a200_bn_svhn",
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
        (pyramidnet110_a48_cifar10, 10),
        (pyramidnet110_a48_cifar100, 100),
        (pyramidnet110_a48_svhn, 10),
        (pyramidnet110_a84_cifar10, 10),
        (pyramidnet110_a84_cifar100, 100),
        (pyramidnet110_a84_svhn, 10),
        (pyramidnet110_a270_cifar10, 10),
        (pyramidnet110_a270_cifar100, 100),
        (pyramidnet110_a270_svhn, 10),
        (pyramidnet164_a270_bn_cifar10, 10),
        (pyramidnet164_a270_bn_cifar100, 100),
        (pyramidnet164_a270_bn_svhn, 10),
        (pyramidnet200_a240_bn_cifar10, 10),
        (pyramidnet200_a240_bn_cifar100, 100),
        (pyramidnet200_a240_bn_svhn, 10),
        (pyramidnet236_a220_bn_cifar10, 10),
        (pyramidnet236_a220_bn_cifar100, 100),
        (pyramidnet236_a220_bn_svhn, 10),
        (pyramidnet272_a200_bn_cifar10, 10),
        (pyramidnet272_a200_bn_cifar100, 100),
        (pyramidnet272_a200_bn_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained, num_classes=num_classes)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pyramidnet110_a48_cifar10 or weight_count == 1772706)
        assert (model != pyramidnet110_a48_cifar100 or weight_count == 1778556)
        assert (model != pyramidnet110_a48_svhn or weight_count == 1772706)
        assert (model != pyramidnet110_a84_cifar10 or weight_count == 3904446)
        assert (model != pyramidnet110_a84_cifar100 or weight_count == 3913536)
        assert (model != pyramidnet110_a84_svhn or weight_count == 3904446)
        assert (model != pyramidnet110_a270_cifar10 or weight_count == 28485477)
        assert (model != pyramidnet110_a270_cifar100 or weight_count == 28511307)
        assert (model != pyramidnet110_a270_svhn or weight_count == 28485477)
        assert (model != pyramidnet164_a270_bn_cifar10 or weight_count == 27216021)
        assert (model != pyramidnet164_a270_bn_cifar100 or weight_count == 27319071)
        assert (model != pyramidnet164_a270_bn_svhn or weight_count == 27216021)
        assert (model != pyramidnet200_a240_bn_cifar10 or weight_count == 26752702)
        assert (model != pyramidnet200_a240_bn_cifar100 or weight_count == 26844952)
        assert (model != pyramidnet200_a240_bn_svhn or weight_count == 26752702)
        assert (model != pyramidnet236_a220_bn_cifar10 or weight_count == 26969046)
        assert (model != pyramidnet236_a220_bn_cifar100 or weight_count == 27054096)
        assert (model != pyramidnet236_a220_bn_svhn or weight_count == 26969046)
        assert (model != pyramidnet272_a200_bn_cifar10 or weight_count == 26210842)
        assert (model != pyramidnet272_a200_bn_cifar100 or weight_count == 26288692)
        assert (model != pyramidnet272_a200_bn_svhn or weight_count == 26210842)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
