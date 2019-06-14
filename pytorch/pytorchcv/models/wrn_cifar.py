"""
    WRN for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['CIFARWRN', 'wrn16_10_cifar10', 'wrn16_10_cifar100', 'wrn16_10_svhn', 'wrn28_10_cifar10',
           'wrn28_10_cifar100', 'wrn28_10_svhn', 'wrn40_8_cifar10', 'wrn40_8_cifar100', 'wrn40_8_svhn']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3
from .preresnet import PreResUnit, PreResActivation


class CIFARWRN(nn.Module):
    """
    WRN model for CIFAR from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
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
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARWRN, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), PreResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=False,
                    conv1_stride=False))
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


def get_wrn_cifar(num_classes,
                  blocks,
                  width_factor,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create WRN model for CIFAR with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    width_factor : int
        Wide scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    assert ((blocks - 4) % 6 == 0)
    layers = [(blocks - 4) // 6] * 3
    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci * width_factor] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFARWRN(
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


def wrn16_10_cifar10(num_classes=10, **kwargs):
    """
    WRN-16-10 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=16, width_factor=10, model_name="wrn16_10_cifar10", **kwargs)


def wrn16_10_cifar100(num_classes=100, **kwargs):
    """
    WRN-16-10 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=16, width_factor=10, model_name="wrn16_10_cifar100", **kwargs)


def wrn16_10_svhn(num_classes=10, **kwargs):
    """
    WRN-16-10 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=16, width_factor=10, model_name="wrn16_10_svhn", **kwargs)


def wrn28_10_cifar10(num_classes=10, **kwargs):
    """
    WRN-28-10 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=28, width_factor=10, model_name="wrn28_10_cifar10", **kwargs)


def wrn28_10_cifar100(num_classes=100, **kwargs):
    """
    WRN-28-10 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=28, width_factor=10, model_name="wrn28_10_cifar100", **kwargs)


def wrn28_10_svhn(num_classes=10, **kwargs):
    """
    WRN-28-10 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=28, width_factor=10, model_name="wrn28_10_svhn", **kwargs)


def wrn40_8_cifar10(num_classes=10, **kwargs):
    """
    WRN-40-8 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=40, width_factor=8, model_name="wrn40_8_cifar10", **kwargs)


def wrn40_8_cifar100(num_classes=100, **kwargs):
    """
    WRN-40-8 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=40, width_factor=8, model_name="wrn40_8_cifar100", **kwargs)


def wrn40_8_svhn(num_classes=10, **kwargs):
    """
    WRN-40-8 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(num_classes=num_classes, blocks=40, width_factor=8, model_name="wrn40_8_svhn", **kwargs)


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
        (wrn16_10_cifar10, 10),
        (wrn16_10_cifar100, 100),
        (wrn16_10_svhn, 10),
        (wrn28_10_cifar10, 10),
        (wrn28_10_cifar100, 100),
        (wrn28_10_svhn, 10),
        (wrn40_8_cifar10, 10),
        (wrn40_8_cifar100, 100),
        (wrn40_8_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn16_10_cifar10 or weight_count == 17116634)
        assert (model != wrn16_10_cifar100 or weight_count == 17174324)
        assert (model != wrn16_10_svhn or weight_count == 17116634)
        assert (model != wrn28_10_cifar10 or weight_count == 36479194)
        assert (model != wrn28_10_cifar100 or weight_count == 36536884)
        assert (model != wrn28_10_svhn or weight_count == 36479194)
        assert (model != wrn40_8_cifar10 or weight_count == 35748314)
        assert (model != wrn40_8_cifar100 or weight_count == 35794484)
        assert (model != wrn40_8_svhn or weight_count == 35748314)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
