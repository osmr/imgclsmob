"""
    Single-Path NASNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.
"""

__all__ = ['SPNASNet', 'spnasnet']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block


class SPNASUnit(nn.Module):
    """
    Single-Path NASNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int
        Expansion factor for each unit.
    use_skip : bool, default True
        Whether to use skip connection.
    activation : str, default 'relu'
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_kernel3,
                 exp_factor,
                 use_skip=True,
                 activation="relu"):
        super(SPNASUnit, self).__init__()
        assert (exp_factor >= 1)
        self.residual = (in_channels == out_channels) and (stride == 1) and use_skip
        self.use_exp_conv = exp_factor > 1
        mid_channels = exp_factor * in_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation)
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class SPNASInitBlock(nn.Module):
    """
    Single-Path NASNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(SPNASInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2)
        self.conv2 = SPNASUnit(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1,
            use_kernel3=True,
            exp_factor=1,
            use_skip=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SPNASFinalBlock(nn.Module):
    """
    Single-Path NASNet specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(SPNASFinalBlock, self).__init__()
        self.conv1 = SPNASUnit(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=1,
            use_kernel3=True,
            exp_factor=6,
            use_skip=False)
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SPNASNet(nn.Module):
    """
    Single-Path NASNet model from 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Number of output channels for the initial unit.
    final_block_channels : list of 2 int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
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
                 kernels3,
                 exp_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(SPNASNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", SPNASInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels[1],
            mid_channels=init_block_channels[0]))
        in_channels = init_block_channels[1]
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if ((j == 0) and (i != 3)) or ((j == len(channels_per_stage) // 2) and (i == 3)) else 1
                use_kernel3 = kernels3[i][j] == 1
                exp_factor = exp_factors[i][j]
                stage.add_module("unit{}".format(j + 1), SPNASUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", SPNASFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels[1],
            mid_channels=final_block_channels[0]))
        in_channels = final_block_channels[1]
        self.features.add_module("final_pool", nn.AvgPool2d(
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


def get_spnasnet(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".torch", "models"),
                 **kwargs):
    """
    Create Single-Path NASNet model with specific parameters.

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
    final_block_channels = [320, 1280]
    channels = [[24, 24, 24], [40, 40, 40, 40], [80, 80, 80, 80], [96, 96, 96, 96, 192, 192, 192, 192]]
    kernels3 = [[1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]
    exp_factors = [[3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3, 6, 6, 6, 6]]

    net = SPNASNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
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


def spnasnet(**kwargs):
    """
    Single-Path NASNet model from 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_spnasnet(model_name="spnasnet", **kwargs)


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
        spnasnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != spnasnet or weight_count == 4421616)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
