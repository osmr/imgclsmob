"""
    ESPNetv2 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.
"""

__all__ = ['ESPNetv2', 'espnetv2_wd2', 'espnetv2_w1', 'espnetv2_w5d4', 'espnetv2_w3d2', 'espnetv2_w2']

import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3, conv1x1_block, conv3x3_block, DualPathSequential


class PreActivation(nn.Module):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PreActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ShortcutBlock(nn.Module):
    """
    ESPNetv2 shortcut block.

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
        super(ShortcutBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            activation=(lambda: nn.PReLU(in_channels)))
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HierarchicalConcurrent(nn.Sequential):
    """
    A container for hierarchical concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=1):
        super(HierarchicalConcurrent, self).__init__()
        self.axis = axis

    def forward(self, x):
        out = []
        y_prev = None
        for module in self._modules.values():
            y = module(x)
            if y_prev is not None:
                y += y_prev
            out.append(y)
            y_prev = y
        out = torch.cat(tuple(out), dim=self.axis)
        return out


class ESPBlock(nn.Module):
    """
    ESPNetv2 block (so-called EESP block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    dilations : list of int
        Dilation values for branches.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilations):
        super(ESPBlock, self).__init__()
        num_branches = len(dilations)
        assert (out_channels % num_branches == 0)
        self.downsample = (stride != 1)
        mid_channels = out_channels // num_branches

        self.reduce_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=num_branches,
            activation=(lambda: nn.PReLU(mid_channels)))

        self.branches = HierarchicalConcurrent()
        for i in range(num_branches):
            self.branches.add_module("branch{}".format(i + 1), conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                padding=dilations[i],
                dilation=dilations[i],
                groups=mid_channels))

        self.merge_conv = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=num_branches,
            activation=None)
        self.preactiv = PreActivation(in_channels=out_channels)
        self.activ = nn.PReLU(out_channels)

    def forward(self, x, x0):
        y = self.reduce_conv(x)
        y = self.branches(y)
        y = self.preactiv(y)
        y = self.merge_conv(y)
        if not self.downsample:
            y = y + x
            y = self.activ(y)
        return y, x0


class DownsampleBlock(nn.Module):
    """
    ESPNetv2 downsample block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    x0_channels : int
        Number of input channels for shortcut.
    dilations : list of int
        Dilation values for branches in EESP block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 x0_channels,
                 dilations):
        super(DownsampleBlock, self).__init__()
        inc_channels = out_channels - in_channels

        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1)
        self.eesp = ESPBlock(
            in_channels=in_channels,
            out_channels=inc_channels,
            stride=2,
            dilations=dilations)
        self.shortcut_block = ShortcutBlock(
            in_channels=x0_channels,
            out_channels=out_channels)
        self.activ = nn.PReLU(out_channels)

    def forward(self, x, x0):
        y1 = self.pool(x)
        y2, _ = self.eesp(x, None)
        x = torch.cat((y1, y2), dim=1)
        x0 = self.pool(x0)
        y3 = self.shortcut_block(x0)
        x = x + y3
        x = self.activ(x)
        return x, x0


class ESPInitBlock(nn.Module):
    """
    ESPNetv2 initial block.

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
        super(ESPInitBlock, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            activation=(lambda: nn.PReLU(out_channels)))
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x, x0):
        x = self.conv(x)
        x0 = self.pool(x0)
        return x, x0


class ESPFinalBlock(nn.Module):
    """
    ESPNetv2 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    final_groups : int
        Number of groups in the last convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 final_groups):
        super(ESPFinalBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            activation=(lambda: nn.PReLU(in_channels)))
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=final_groups,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ESPNetv2(nn.Module):
    """
    ESPNetv2 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    final_block_groups : int
        Number of groups for the final unit.
    dilations : list of list of list of int
        Dilation values for branches in each unit.
    dropout_rate : float, default 0.2
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 final_block_groups,
                 dilations,
                 dropout_rate=0.2,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ESPNetv2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        x0_channels = in_channels

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=0,
            last_ordinals=2)
        self.features.add_module("init_block", ESPInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    unit = DownsampleBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        x0_channels=x0_channels,
                        dilations=dilations[i][j])
                else:
                    unit = ESPBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        dilations=dilations[i][j])
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", ESPFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            final_groups=final_block_groups))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Sequential()
        self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=in_channels,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x, x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_espnetv2(width_scale,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".torch", "models"),
                 **kwargs):
    """
    Create ESPNetv2 model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    assert (width_scale <= 2.0)

    branches = 4
    layers = [1, 4, 8, 4]

    max_dilation_list = [6, 5, 4, 3, 2]
    max_dilations = [[max_dilation_list[i]] + [max_dilation_list[i + 1]] * (li - 1) for (i, li) in enumerate(layers)]
    dilations = [[sorted([k + 1 if k < dij else 1 for k in range(branches)]) for dij in di] for di in max_dilations]

    base_channels = 32
    weighed_base_channels = math.ceil(float(math.floor(base_channels * width_scale)) / branches) * branches
    channels_per_layers = [weighed_base_channels * pow(2, i + 1) for i in range(len(layers))]

    init_block_channels = base_channels if weighed_base_channels > base_channels else weighed_base_channels
    final_block_channels = 1024 if width_scale <= 1.5 else 1280

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ESPNetv2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        final_block_groups=branches,
        dilations=dilations,
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


def espnetv2_wd2(**kwargs):
    """
    ESPNetv2 x0.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=0.5, model_name="espnetv2_wd2", **kwargs)


def espnetv2_w1(**kwargs):
    """
    ESPNetv2 x1.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.0, model_name="espnetv2_w1", **kwargs)


def espnetv2_w5d4(**kwargs):
    """
    ESPNetv2 x1.25 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.25, model_name="espnetv2_w5d4", **kwargs)


def espnetv2_w3d2(**kwargs):
    """
    ESPNetv2 x1.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.5, model_name="espnetv2_w3d2", **kwargs)


def espnetv2_w2(**kwargs):
    """
    ESPNetv2 x2.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=2.0, model_name="espnetv2_w2", **kwargs)


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
        espnetv2_wd2,
        espnetv2_w1,
        espnetv2_w5d4,
        espnetv2_w3d2,
        espnetv2_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espnetv2_wd2 or weight_count == 1241332)
        assert (model != espnetv2_w1 or weight_count == 1670072)
        assert (model != espnetv2_w5d4 or weight_count == 1965440)
        assert (model != espnetv2_w3d2 or weight_count == 2314856)
        assert (model != espnetv2_w2 or weight_count == 3498136)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
