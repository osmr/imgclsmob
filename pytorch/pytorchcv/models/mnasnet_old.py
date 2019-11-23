"""
    MnasNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,' https://arxiv.org/abs/1807.11626.
"""

__all__ = ['MnasNet', 'mnasnet']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block


class DwsConvBlock(nn.Module):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
        super(DwsConvBlock, self).__init__()
        self.dw_conv = dwconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MnasUnit(nn.Module):
    """
    MnasNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    expansion_factor : int
        Factor for expansion of channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expansion_factor):
        super(MnasUnit, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * expansion_factor
        dwconv_block_fn = dwconv3x3_block if kernel_size == 3 else (dwconv5x5_block if kernel_size == 5 else None)

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = dwconv_block_fn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class MnasInitBlock(nn.Module):
    """
    MnasNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of 2 int
        Numbers of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(MnasInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            stride=2)
        self.conv2 = DwsConvBlock(
            in_channels=out_channels_list[0],
            out_channels=out_channels_list[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MnasNet(nn.Module):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Numbers of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list of list of int
        Number of kernel sizes for each unit.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
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
                 final_block_channels,
                 kernel_sizes,
                 expansion_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MnasNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", MnasInitBlock(
            in_channels=in_channels,
            out_channels_list=init_block_channels))
        in_channels = init_block_channels[-1]
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                stride = 2 if (j == 0) else 1
                stage.add_module("unit{}".format(j + 1), MnasUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    expansion_factor=expansion_factor))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels))
        in_channels = final_block_channels
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


def get_mnasnet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create MnasNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    init_block_channels = [32, 16]
    final_block_channels = 1280
    layers = [3, 3, 3, 2, 4, 1]
    downsample = [1, 1, 1, 0, 1, 0]
    channels_per_layers = [24, 40, 80, 96, 192, 320]
    expansion_factors_per_layers = [3, 3, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 5, 5, 3, 5, 3]
    default_kernel_size = 3

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] + [default_kernel_size] * (y[1] - 1)] if y[2] != 0 else x[:-1] + [
        x[-1] + [y[0]] + [default_kernel_size] * (y[1] - 1)], zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])

    net = MnasNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        expansion_factors=expansion_factors,
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


def mnasnet(**kwargs):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(model_name="mnasnet", **kwargs)


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
        mnasnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mnasnet or weight_count == 4308816)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
