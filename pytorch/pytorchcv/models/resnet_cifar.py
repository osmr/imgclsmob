"""
    ResNet for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['CIFARResNet', 'resnet20_cifar10', 'resnet20_cifar100', 'resnet20_svhn', 'resnet56_cifar10',
           'resnet56_cifar100', 'resnet56_svhn', 'resnet110_cifar10', 'resnet110_cifar100', 'resnet110_svhn',
           'resnet164bn_cifar10', 'resnet164bn_cifar100', 'resnet164bn_svhn', 'resnet1001_cifar10',
           'resnet1001_cifar100', 'resnet1001_svhn', 'resnet1202_cifar10', 'resnet1202_cifar100', 'resnet1202_svhn']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3_block
from .resnet import ResUnit


class CIFARResNet(nn.Module):
    """
    ResNet model for CIFAR from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

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
        super(CIFARResNet, self).__init__()
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
                stage.add_module("unit{}".format(j + 1), ResUnit(
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


def get_resnet_cifar(num_classes,
                     blocks,
                     bottleneck,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".torch", "models"),
                     **kwargs):
    """
    Create ResNet model for CIFAR with specific parameters.

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

    net = CIFARResNet(
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


def resnet20_cifar10(num_classes=10, **kwargs):
    """
    ResNet-20 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False, model_name="resnet20_cifar10",
                            **kwargs)


def resnet20_cifar100(num_classes=100, **kwargs):
    """
    ResNet-20 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False, model_name="resnet20_cifar100",
                            **kwargs)


def resnet20_svhn(num_classes=10, **kwargs):
    """
    ResNet-20 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=20, bottleneck=False, model_name="resnet20_svhn",
                            **kwargs)


def resnet56_cifar10(num_classes=10, **kwargs):
    """
    ResNet-56 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False, model_name="resnet56_cifar10",
                            **kwargs)


def resnet56_cifar100(num_classes=100, **kwargs):
    """
    ResNet-56 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False, model_name="resnet56_cifar100",
                            **kwargs)


def resnet56_svhn(num_classes=10, **kwargs):
    """
    ResNet-56 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=56, bottleneck=False, model_name="resnet56_svhn",
                            **kwargs)


def resnet110_cifar10(num_classes=10, **kwargs):
    """
    ResNet-110 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False, model_name="resnet110_cifar10",
                            **kwargs)


def resnet110_cifar100(num_classes=100, **kwargs):
    """
    ResNet-110 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False, model_name="resnet110_cifar100",
                            **kwargs)


def resnet110_svhn(num_classes=10, **kwargs):
    """
    ResNet-110 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=110, bottleneck=False, model_name="resnet110_svhn",
                            **kwargs)


def resnet164bn_cifar10(num_classes=10, **kwargs):
    """
    ResNet-164(BN) model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True, model_name="resnet164bn_cifar10",
                            **kwargs)


def resnet164bn_cifar100(num_classes=100, **kwargs):
    """
    ResNet-164(BN) model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True, model_name="resnet164bn_cifar100",
                            **kwargs)


def resnet164bn_svhn(num_classes=10, **kwargs):
    """
    ResNet-164(BN) model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=164, bottleneck=True, model_name="resnet164bn_svhn",
                            **kwargs)


def resnet1001_cifar10(num_classes=10, **kwargs):
    """
    ResNet-1001 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True, model_name="resnet1001_cifar10",
                            **kwargs)


def resnet1001_cifar100(num_classes=100, **kwargs):
    """
    ResNet-1001 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True, model_name="resnet1001_cifar100",
                            **kwargs)


def resnet1001_svhn(num_classes=10, **kwargs):
    """
    ResNet-1001 model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=1001, bottleneck=True, model_name="resnet1001_svhn",
                            **kwargs)


def resnet1202_cifar10(num_classes=10, **kwargs):
    """
    ResNet-1202 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False, model_name="resnet1202_cifar10",
                            **kwargs)


def resnet1202_cifar100(num_classes=100, **kwargs):
    """
    ResNet-1202 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False, model_name="resnet1202_cifar100",
                            **kwargs)


def resnet1202_svhn(num_classes=10, **kwargs):
    """
    ResNet-1202 model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(num_classes=num_classes, blocks=1202, bottleneck=False, model_name="resnet1202_svhn",
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
        (resnet20_cifar10, 10),
        (resnet20_cifar100, 100),
        (resnet20_svhn, 10),
        (resnet56_cifar10, 10),
        (resnet56_cifar100, 100),
        (resnet56_svhn, 10),
        (resnet110_cifar10, 10),
        (resnet110_cifar100, 100),
        (resnet110_svhn, 10),
        (resnet164bn_cifar10, 10),
        (resnet164bn_cifar100, 100),
        (resnet164bn_svhn, 10),
        (resnet1001_cifar10, 10),
        (resnet1001_cifar100, 100),
        (resnet1001_svhn, 10),
        (resnet1202_cifar10, 10),
        (resnet1202_cifar100, 100),
        (resnet1202_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnet20_cifar10 or weight_count == 272474)
        assert (model != resnet20_cifar100 or weight_count == 278324)
        assert (model != resnet20_svhn or weight_count == 272474)
        assert (model != resnet56_cifar10 or weight_count == 855770)
        assert (model != resnet56_cifar100 or weight_count == 861620)
        assert (model != resnet56_svhn or weight_count == 855770)
        assert (model != resnet110_cifar10 or weight_count == 1730714)
        assert (model != resnet110_cifar100 or weight_count == 1736564)
        assert (model != resnet110_svhn or weight_count == 1730714)
        assert (model != resnet164bn_cifar10 or weight_count == 1704154)
        assert (model != resnet164bn_cifar100 or weight_count == 1727284)
        assert (model != resnet164bn_svhn or weight_count == 1704154)
        assert (model != resnet1001_cifar10 or weight_count == 10328602)
        assert (model != resnet1001_cifar100 or weight_count == 10351732)
        assert (model != resnet1001_svhn or weight_count == 10328602)
        assert (model != resnet1202_cifar10 or weight_count == 19424026)
        assert (model != resnet1202_cifar100 or weight_count == 19429876)
        assert (model != resnet1202_svhn or weight_count == 19424026)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
