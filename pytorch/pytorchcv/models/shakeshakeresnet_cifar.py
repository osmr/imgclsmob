"""
    Shake-Shake-ResNet for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.
"""

__all__ = ['CIFARShakeShakeResNet', 'shakeshakeresnet20_2x16d_cifar10', 'shakeshakeresnet20_2x16d_cifar100',
           'shakeshakeresnet20_2x16d_svhn', 'shakeshakeresnet26_2x32d_cifar10', 'shakeshakeresnet26_2x32d_cifar100',
           'shakeshakeresnet26_2x32d_svhn']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, conv3x3_block
from .resnet import ResBlock, ResBottleneck


class ShakeShake(torch.autograd.Function):
    """
    Shake-Shake function.
    """

    @staticmethod
    def forward(ctx, x1, x2, alpha):
        y = alpha * x1 + (1 - alpha) * x2
        return y

    @staticmethod
    def backward(ctx, dy):
        beta = torch.rand(dy.size(0), dtype=dy.dtype, device=dy.device).view(-1, 1, 1, 1)
        return beta * dy, (1 - beta) * dy, None


class ShakeShakeShortcut(nn.Module):
    """
    Shake-Shake-ResNet shortcut.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(ShakeShakeShortcut, self).__init__()
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        self.pool = nn.AvgPool2d(
            kernel_size=1,
            stride=stride)
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        x1 = self.pool(x)
        x1 = self.conv1(x1)
        x2 = x[:, :, :-1, :-1].contiguous()
        x2 = self.pad(x2)
        x2 = self.pool(x2)
        x2 = self.conv2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        return x


class ShakeShakeResUnit(nn.Module):
    """
    Shake-Shake-ResNet unit with residual connection.

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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(ShakeShakeResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        branch_class = ResBottleneck if bottleneck else ResBlock

        self.branch1 = branch_class(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.branch2 = branch_class(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        if self.resize_identity:
            self.identity_branch = ShakeShakeShortcut(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        self.activ = nn.ReLU(inplace=True)
        self.shake_shake = ShakeShake.apply

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_branch(x)
        else:
            identity = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        if self.training:
            alpha = torch.rand(x1.size(0), dtype=x1.dtype, device=x1.device).view(-1, 1, 1, 1)
            x = self.shake_shake(x1, x2, alpha)
        else:
            x = 0.5 * (x1 + x2)
        x = x + identity
        x = self.activ(x)
        return x


class CIFARShakeShakeResNet(nn.Module):
    """
    Shake-Shake-ResNet model for CIFAR from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

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
        super(CIFARShakeShakeResNet, self).__init__()
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
                stage.add_module("unit{}".format(j + 1), ShakeShakeResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck))
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


def get_shakeshakeresnet_cifar(classes,
                               blocks,
                               bottleneck,
                               first_stage_channels=16,
                               model_name=None,
                               pretrained=False,
                               root=os.path.join("~", ".torch", "models"),
                               **kwargs):
    """
    Create Shake-Shake-ResNet model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    first_stage_channels : int, default 16
        Number of output channels for the first stage.
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

    from functools import reduce
    channels_per_layers = reduce(lambda x, y: x + [x[-1] * 2], range(2), [first_stage_channels])

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = CIFARShakeShakeResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
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


def shakeshakeresnet20_2x16d_cifar10(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-20-2x16d model for CIFAR-10 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, first_stage_channels=16,
                                      model_name="shakeshakeresnet20_2x16d_cifar10", **kwargs)


def shakeshakeresnet20_2x16d_cifar100(classes=100, **kwargs):
    """
    Shake-Shake-ResNet-20-2x16d model for CIFAR-100 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, first_stage_channels=16,
                                      model_name="shakeshakeresnet20_2x16d_cifar100", **kwargs)


def shakeshakeresnet20_2x16d_svhn(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-20-2x16d model for SVHN from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, first_stage_channels=16,
                                      model_name="shakeshakeresnet20_2x16d_svhn", **kwargs)


def shakeshakeresnet26_2x32d_cifar10(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-26-2x32d model for CIFAR-10 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=26, bottleneck=False, first_stage_channels=32,
                                      model_name="shakeshakeresnet26_2x32d_cifar10", **kwargs)


def shakeshakeresnet26_2x32d_cifar100(classes=100, **kwargs):
    """
    Shake-Shake-ResNet-26-2x32d model for CIFAR-100 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=26, bottleneck=False, first_stage_channels=32,
                                      model_name="shakeshakeresnet26_2x32d_cifar100", **kwargs)


def shakeshakeresnet26_2x32d_svhn(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-26-2x32d model for SVHN from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=26, bottleneck=False, first_stage_channels=32,
                                      model_name="shakeshakeresnet26_2x32d_svhn", **kwargs)


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
        (shakeshakeresnet20_2x16d_cifar10, 10),
        (shakeshakeresnet20_2x16d_cifar100, 100),
        (shakeshakeresnet20_2x16d_svhn, 10),
        (shakeshakeresnet26_2x32d_cifar10, 10),
        (shakeshakeresnet26_2x32d_cifar100, 100),
        (shakeshakeresnet26_2x32d_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shakeshakeresnet20_2x16d_cifar10 or weight_count == 541082)
        assert (model != shakeshakeresnet20_2x16d_cifar100 or weight_count == 546932)
        assert (model != shakeshakeresnet20_2x16d_svhn or weight_count == 541082)
        assert (model != shakeshakeresnet26_2x32d_cifar10 or weight_count == 2923162)
        assert (model != shakeshakeresnet26_2x32d_cifar100 or weight_count == 2934772)
        assert (model != shakeshakeresnet26_2x32d_svhn or weight_count == 2923162)

        x = torch.randn(14, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, num_classes))


if __name__ == "__main__":
    _test()
