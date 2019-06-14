"""
    X-DenseNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.
"""

__all__ = ['XDenseNet', 'xdensenet121_2', 'xdensenet161_2', 'xdensenet169_2', 'xdensenet201_2', 'pre_xconv3x3_block',
           'XDenseUnit']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .preresnet import PreResInitBlock, PreResActivation
from .densenet import TransitionBlock


class XConv2d(nn.Conv2d):
    """
    X-Convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    groups : int, default 1
        Number of groups.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups=1,
                 expand_ratio=2,
                 **kwargs):
        super(XConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            **kwargs)
        self.expand_ratio = expand_ratio
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        grouped_in_channels = in_channels // groups
        self.mask = torch.nn.Parameter(
            data=torch.Tensor(out_channels, grouped_in_channels, *kernel_size),
            requires_grad=False)
        self.init_parameters()

    def init_parameters(self):
        shape = self.mask.shape
        expand_size = max(shape[1] // self.expand_ratio, 1)
        self.mask[:] = 0
        for i in range(shape[0]):
            jj = torch.randperm(shape[1], device=self.mask.device)[:expand_size]
            self.mask[i, jj, :, :] = 1

    def forward(self, input):
        masked_weight = self.weight.mul(self.mask)
        return F.conv2d(
            input=input,
            weight=masked_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)


class PreXConvBlock(nn.Module):
    """
    X-Convolution block with Batch normalization and ReLU pre-activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 return_preact=False,
                 activate=True,
                 expand_ratio=2):
        super(PreXConvBlock, self).__init__()
        self.return_preact = return_preact
        self.activate = activate

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)
        self.conv = XConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            expand_ratio=expand_ratio)

    def forward(self, x):
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_xconv1x1_block(in_channels,
                       out_channels,
                       stride=1,
                       bias=False,
                       return_preact=False,
                       activate=True,
                       expand_ratio=2):
    """
    1x1 version of the pre-activated x-convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    return PreXConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
        return_preact=return_preact,
        activate=activate,
        expand_ratio=expand_ratio)


def pre_xconv3x3_block(in_channels,
                       out_channels,
                       stride=1,
                       padding=1,
                       dilation=1,
                       return_preact=False,
                       activate=True,
                       expand_ratio=2):
    """
    3x3 version of the pre-activated x-convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    return PreXConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_preact=return_preact,
        activate=activate,
        expand_ratio=expand_ratio)


class XDenseUnit(nn.Module):
    """
    X-DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    expand_ratio : int
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate,
                 expand_ratio):
        super(XDenseUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        self.conv1 = pre_xconv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            expand_ratio=expand_ratio)
        self.conv2 = pre_xconv3x3_block(
            in_channels=mid_channels,
            out_channels=inc_channels,
            expand_ratio=expand_ratio)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.cat((identity, x), dim=1)
        return x


class XDenseNet(nn.Module):
    """
    X-DenseNet model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    expand_ratio : int, default 2
        Ratio of expansion.
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
                 dropout_rate=0.0,
                 expand_ratio=2,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(XDenseNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", PreResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            if i != 0:
                stage.add_module("trans{}".format(i + 1), TransitionBlock(
                    in_channels=in_channels,
                    out_channels=(in_channels // 2)))
                in_channels = in_channels // 2
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), XDenseUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    expand_ratio=expand_ratio))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("post_activ", PreResActivation(in_channels=in_channels))
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


def get_xdensenet(blocks,
                  expand_ratio=2,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create X-DenseNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    expand_ratio : int, default 2
        Ratio of expansion.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif blocks == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif blocks == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif blocks == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported X-DenseNet version with number of layers {}".format(blocks))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = XDenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        expand_ratio=expand_ratio,
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


def xdensenet121_2(**kwargs):
    """
    X-DenseNet-121-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=121, model_name="xdensenet121_2", **kwargs)


def xdensenet161_2(**kwargs):
    """
    X-DenseNet-161-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=161, model_name="xdensenet161_2", **kwargs)


def xdensenet169_2(**kwargs):
    """
    X-DenseNet-169-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=169, model_name="xdensenet169_2", **kwargs)


def xdensenet201_2(**kwargs):
    """
    X-DenseNet-201-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=201, model_name="xdensenet201_2", **kwargs)


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
        xdensenet121_2,
        xdensenet161_2,
        xdensenet169_2,
        xdensenet201_2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xdensenet121_2 or weight_count == 7978856)
        assert (model != xdensenet161_2 or weight_count == 28681000)
        assert (model != xdensenet169_2 or weight_count == 14149480)
        assert (model != xdensenet201_2 or weight_count == 20013928)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
