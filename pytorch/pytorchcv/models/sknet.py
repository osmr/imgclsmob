"""
    SKNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.
"""

__all__ = ['SKNet', 'sknet50', 'sknet101', 'sknet152']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent
from .resnet import ResInitBlock


class SKConvBlock(nn.Module):
    """
    SKNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int, default 32
        Number of groups in branches.
    num_branches : int, default 2
        Number of branches (`M` parameter in the paper).
    reduction : int, default 16
        Reduction value for intermediate channels (`r` parameter in the paper).
    min_channels : int, default 32
        Minimal number of intermediate channels (`L` parameter in the paper).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=32,
                 num_branches=2,
                 reduction=16,
                 min_channels=32):
        super(SKConvBlock, self).__init__()
        self.num_branches = num_branches
        self.out_channels = out_channels
        mid_channels = max(in_channels // reduction, min_channels)

        self.branches = Concurrent(stack=True)
        for i in range(num_branches):
            dilation = 1 + i
            self.branches.add_module("branch{}".format(i + 2), conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=groups))
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = conv1x1_block(
            in_channels=out_channels,
            out_channels=mid_channels)
        self.fc2 = conv1x1(
            in_channels=mid_channels,
            out_channels=(out_channels * num_branches))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.branches(x)

        u = y.sum(dim=1)
        s = self.pool(u)
        z = self.fc1(s)
        w = self.fc2(z)

        batch = w.size(0)
        w = w.view(batch, self.num_branches, self.out_channels)
        w = self.softmax(w)
        w = w.unsqueeze(-1).unsqueeze(-1)

        y = y * w
        y = y.sum(dim=1)
        return y


class SKNetBottleneck(nn.Module):
    """
    SKNet bottleneck block for residual path in SKNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_factor=2):
        super(SKNetBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = SKConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SKNetUnit(nn.Module):
    """
    SKNet unit.

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
        super(SKNetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = SKNetBottleneck(
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
        x = x + identity
        x = self.activ(x)
        return x


class SKNet(nn.Module):
    """
    SKNet model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

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
        super(SKNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SKNetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
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


def get_sknet(blocks,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create SKNet model with specific parameters.

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

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported SKNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SKNet(
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


def sknet50(**kwargs):
    """
    SKNet-50 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=50, model_name="sknet50", **kwargs)


def sknet101(**kwargs):
    """
    SKNet-101 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=101, model_name="sknet101", **kwargs)


def sknet152(**kwargs):
    """
    SKNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=152, model_name="sknet152", **kwargs)


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
        sknet50,
        sknet101,
        sknet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sknet50 or weight_count == 27479784)
        assert (model != sknet101 or weight_count == 48736040)
        assert (model != sknet152 or weight_count == 66295656)

        x = torch.randn(14, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, 1000))


if __name__ == "__main__":
    _test()
