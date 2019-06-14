"""
    SparseNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Sparsely Aggregated Convolutional Networks,' https://arxiv.org/abs/1801.05895.
"""

__all__ = ['SparseNet', 'sparsenet121', 'sparsenet161', 'sparsenet169', 'sparsenet201', 'sparsenet264']

import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import pre_conv1x1_block, pre_conv3x3_block
from .preresnet import PreResInitBlock, PreResActivation
from .densenet import TransitionBlock


def sparsenet_exponential_fetch(lst):
    """
    SparseNet's specific exponential fetch.

    Parameters:
    ----------
    lst : list
        List of something.

    Returns
    -------
    list
        Filtered list.
    """
    return [lst[len(lst) - 2**i] for i in range(1 + math.floor(math.log(len(lst), 2)))]


class SparseBlock(nn.Module):
    """
    SparseNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate):
        super(SparseBlock, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        mid_channels = out_channels * bn_size

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class SparseStage(nn.Module):
    """
    SparseNet stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels_per_stage : list of int
        Number of output channels for each unit in stage.
    growth_rate : int
        Growth rate for blocks.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    do_transition : bool
        Whether use transition block.
    """
    def __init__(self,
                 in_channels,
                 channels_per_stage,
                 growth_rate,
                 dropout_rate,
                 do_transition):
        super(SparseStage, self).__init__()
        self.do_transition = do_transition

        if self.do_transition:
            self.trans = TransitionBlock(
                in_channels=in_channels,
                out_channels=(in_channels // 2))
            in_channels = in_channels // 2
        self.blocks = nn.Sequential()
        for i, out_channels in enumerate(channels_per_stage):
            self.blocks.add_module("block{}".format(i + 1), SparseBlock(
                in_channels=in_channels,
                out_channels=growth_rate,
                dropout_rate=dropout_rate))
            in_channels = out_channels

    def forward(self, x):
        if self.do_transition:
            x = self.trans(x)
        outs = [x]
        for block in self.blocks._modules.values():
            y = block(x)
            outs.append(y)
            flt_outs = sparsenet_exponential_fetch(outs)
            x = torch.cat(tuple(flt_outs), dim=1)
        return x


class SparseNet(nn.Module):
    """
    SparseNet model from 'Sparsely Aggregated Convolutional Networks,' https://arxiv.org/abs/1801.05895.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    growth_rate : int
        Growth rate for blocks.
    dropout_rate : float, default 0.0
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
                 growth_rate,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(SparseNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", PreResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SparseStage(
                in_channels=in_channels,
                channels_per_stage=channels_per_stage,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                do_transition=(i != 0))
            in_channels = channels_per_stage[-1]
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


def get_sparsenet(num_layers,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create SparseNet model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if num_layers == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif num_layers == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif num_layers == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif num_layers == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    elif num_layers == 264:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 64, 48]
    else:
        raise ValueError("Unsupported SparseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [sum(sparsenet_exponential_fetch([xj[0]] + [yj[0]] * (yj[1] + 1)))],
            zip([growth_rate] * yi, range(yi)),
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = SparseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        growth_rate=growth_rate,
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


def sparsenet121(**kwargs):
    """
    SparseNet-121 model from 'Sparsely Aggregated Convolutional Networks,' https://arxiv.org/abs/1801.05895.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sparsenet(num_layers=121, model_name="sparsenet121", **kwargs)


def sparsenet161(**kwargs):
    """
    SparseNet-161 model from 'Sparsely Aggregated Convolutional Networks,' https://arxiv.org/abs/1801.05895.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sparsenet(num_layers=161, model_name="sparsenet161", **kwargs)


def sparsenet169(**kwargs):
    """
    SparseNet-169 model from 'Sparsely Aggregated Convolutional Networks,' https://arxiv.org/abs/1801.05895.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sparsenet(num_layers=169, model_name="sparsenet169", **kwargs)


def sparsenet201(**kwargs):
    """
    SparseNet-201 model from 'Sparsely Aggregated Convolutional Networks,' https://arxiv.org/abs/1801.05895.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sparsenet(num_layers=201, model_name="sparsenet201", **kwargs)


def sparsenet264(**kwargs):
    """
    SparseNet-264 model from 'Sparsely Aggregated Convolutional Networks,' https://arxiv.org/abs/1801.05895.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sparsenet(num_layers=264, model_name="sparsenet264", **kwargs)


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
        sparsenet121,
        sparsenet161,
        sparsenet169,
        sparsenet201,
        sparsenet264,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sparsenet121 or weight_count == 3250824)
        assert (model != sparsenet161 or weight_count == 9853288)
        assert (model != sparsenet169 or weight_count == 4709864)
        assert (model != sparsenet201 or weight_count == 5703144)
        assert (model != sparsenet264 or weight_count == 7717224)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
