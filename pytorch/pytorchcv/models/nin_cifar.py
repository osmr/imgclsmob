"""
    NIN for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Network In Network,' https://arxiv.org/abs/1312.4400.
"""

__all__ = ['CIFARNIN', 'nin_cifar10', 'nin_cifar100', 'nin_svhn']

import os
import torch.nn as nn
import torch.nn.init as init


class NINConv(nn.Module):
    """
    NIN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(NINConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class CIFARNIN(nn.Module):
    """
    NIN model for CIFAR from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_kernel_sizes : list of int
        Convolution window sizes for the first units in each stage.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    num_classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 first_kernel_sizes,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARNIN, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    if i == 1:
                        stage.add_module("pool{}".format(i + 1), nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            padding=1))
                    else:
                        stage.add_module("pool{}".format(i + 1), nn.AvgPool2d(
                            kernel_size=3,
                            stride=2,
                            padding=1))
                    stage.add_module("dropout{}".format(i + 1), nn.Dropout(p=0.5))
                kernel_size = first_kernel_sizes[i] if j == 0 else 1
                padding = (kernel_size - 1) // 2
                stage.add_module("unit{}".format(j + 1), NINConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.output = nn.Sequential()
        self.output.add_module('final_conv', NINConv(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1))
        self.output.add_module('final_pool', nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_nin_cifar(num_classes,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create NIN model for CIFAR with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    channels = [[192, 160, 96], [192, 192, 192], [192, 192]]
    first_kernel_sizes = [5, 5, 3]

    net = CIFARNIN(
        channels=channels,
        first_kernel_sizes=first_kernel_sizes,
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


def nin_cifar10(num_classes=10, **kwargs):
    """
    NIN model for CIFAR-10 from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_nin_cifar(num_classes=num_classes, model_name="nin_cifar10", **kwargs)


def nin_cifar100(num_classes=100, **kwargs):
    """
    NIN model for CIFAR-100 from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_nin_cifar(num_classes=num_classes, model_name="nin_cifar100", **kwargs)


def nin_svhn(num_classes=10, **kwargs):
    """
    NIN model for SVHN from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_nin_cifar(num_classes=num_classes, model_name="nin_svhn", **kwargs)


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
        (nin_cifar10, 10),
        (nin_cifar100, 100),
        (nin_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nin_cifar10 or weight_count == 966986)
        assert (model != nin_cifar100 or weight_count == 984356)
        assert (model != nin_svhn or weight_count == 966986)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
