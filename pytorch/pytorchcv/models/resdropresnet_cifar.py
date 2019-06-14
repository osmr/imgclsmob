"""
    ResDrop-ResNet for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.
"""

__all__ = ['CIFARResDropResNet', 'resdropresnet20_cifar10', 'resdropresnet20_cifar100', 'resdropresnet20_svhn']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block
from .resnet import ResBlock, ResBottleneck


class ResDropResUnit(nn.Module):
    """
    ResDrop-ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    life_prob : float
        Residual branch life probability.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 life_prob):
        super(ResDropResUnit, self).__init__()
        self.life_prob = life_prob
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        body_class = ResBottleneck if bottleneck else ResBlock

        self.body = body_class(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        if self.training:
            b = torch.bernoulli(torch.full((1,), self.life_prob, dtype=x.dtype, device=x.device))
            x = float(b) / self.life_prob * x
        x = x + identity
        x = self.activ(x)
        return x


class CIFARResDropResNet(nn.Module):
    """
    ResDrop-ResNet model for CIFAR from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    life_probs : list of float
        Residual branch life probability for each unit.
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
                 life_probs,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARResDropResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        k = 0
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResDropResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    life_prob=life_probs[k]))
                in_channels = out_channels
                k += 1
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


def get_resdropresnet_cifar(classes,
                            blocks,
                            bottleneck,
                            model_name=None,
                            pretrained=False,
                            root=os.path.join("~", ".torch", "models"),
                            **kwargs):
    """
    Create ResDrop-ResNet model for CIFAR with specific parameters.

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
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    assert (classes in [10, 100])

    if bottleneck:
        assert ((blocks - 2) % 9 == 0)
        layers = [(blocks - 2) // 9] * 3
    else:
        assert ((blocks - 2) % 6 == 0)
        layers = [(blocks - 2) // 6] * 3

    init_block_channels = 16
    channels_per_layers = [16, 32, 64]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    total_layers = sum(layers)
    final_death_prob = 0.5
    life_probs = [1.0 - float(i + 1) / float(total_layers) * final_death_prob for i in range(total_layers)]

    net = CIFARResDropResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        life_probs=life_probs,
        num_classes=classes,
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


def resdropresnet20_cifar10(classes=10, **kwargs):
    """
    ResDrop-ResNet-20 model for CIFAR-10 from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resdropresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resdropresnet20_cifar10",
                                   **kwargs)


def resdropresnet20_cifar100(classes=100, **kwargs):
    """
    ResDrop-ResNet-20 model for CIFAR-100 from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resdropresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resdropresnet20_cifar100",
                                   **kwargs)


def resdropresnet20_svhn(classes=10, **kwargs):
    """
    ResDrop-ResNet-20 model for SVHN from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resdropresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resdropresnet20_svhn",
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
        (resdropresnet20_cifar10, 10),
        (resdropresnet20_cifar100, 100),
        (resdropresnet20_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resdropresnet20_cifar10 or weight_count == 272474)
        assert (model != resdropresnet20_cifar100 or weight_count == 278324)
        assert (model != resdropresnet20_svhn or weight_count == 272474)

        x = torch.randn(14, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, num_classes))


if __name__ == "__main__":
    _test()
