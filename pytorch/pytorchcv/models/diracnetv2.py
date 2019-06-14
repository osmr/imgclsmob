"""
    DiracNetV2 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.
"""

__all__ = ['DiracNetV2', 'diracnet18v2', 'diracnet34v2']

import os
import torch.nn as nn
import torch.nn.init as init


class DiracConv(nn.Module):
    """
    DiracNetV2 specific convolution block with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(DiracConv, self).__init__()
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        return x


def dirac_conv3x3(in_channels,
                  out_channels):
    """
    3x3 version of the DiracNetV2 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DiracConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1)


class DiracInitBlock(nn.Module):
    """
    DiracNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DiracInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DiracNetV2(nn.Module):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DiracNetV2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", DiracInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), dirac_conv3x3(
                    in_channels=in_channels,
                    out_channels=out_channels))
                in_channels = out_channels
            if i != len(channels) - 1:
                stage.add_module("pool{}".format(i + 1), nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                    padding=0))
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_activ', nn.ReLU(inplace=True))
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
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


def get_diracnetv2(blocks,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".torch", "models"),
                   **kwargs):
    """
    Create DiracNetV2 model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if blocks == 18:
        layers = [4, 4, 4, 4]
    elif blocks == 34:
        layers = [6, 8, 12, 6]
    else:
        raise ValueError("Unsupported DiracNetV2 with number of blocks: {}".format(blocks))

    channels_per_layers = [64, 128, 256, 512]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    init_block_channels = 64

    net = DiracNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
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


def diracnet18v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=18, model_name="diracnet18v2", **kwargs)


def diracnet34v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=34, model_name="diracnet34v2", **kwargs)


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
        diracnet18v2,
        diracnet34v2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diracnet18v2 or weight_count == 11511784)
        assert (model != diracnet34v2 or weight_count == 21616232)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
