"""
    VoVNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.
"""

__all__ = ['VoVNet', 'vovnet27s', 'vovnet39', 'vovnet57']

import os
import torch.nn as nn
from .common import conv1x1_block, conv3x3_block, SequentialConcurrent


class VoVUnit(nn.Module):
    """
    VoVNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    branch_channels : int
        Number of output channels for each branch.
    num_branches : int
        Number of branches.
    resize : bool
        Whether to use resize block.
    use_residual : bool
        Whether to use residual block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 branch_channels,
                 num_branches,
                 resize,
                 use_residual):
        super(VoVUnit, self).__init__()
        self.resize = resize
        self.use_residual = use_residual

        if self.resize:
            self.pool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True)

        self.branches = SequentialConcurrent()
        branch_in_channels = in_channels
        for i in range(num_branches):
            self.branches.add_module("branch{}".format(i + 1), conv3x3_block(
                in_channels=branch_in_channels,
                out_channels=branch_channels))
            branch_in_channels = branch_channels

        self.concat_conv = conv1x1_block(
            in_channels=(in_channels + num_branches * branch_channels),
            out_channels=out_channels)

    def forward(self, x):
        if self.resize:
            x = self.pool(x)
        if self.use_residual:
            identity = x
        x = self.branches(x)
        x = self.concat_conv(x)
        if self.use_residual:
            x = x + identity
        return x


class VoVInitBlock(nn.Module):
    """
    VoVNet specific initial block.

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
        super(VoVInitBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class VoVNet(nn.Module):
    """
    VoVNet model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    branch_channels : list of list of int
        Number of branch output channels for each unit.
    num_branches : int
        Number of branches for the each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 branch_channels,
                 num_branches,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(VoVNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        init_block_channels = 128

        self.features = nn.Sequential()
        self.features.add_module("init_block", VoVInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                use_residual = (j != 0)
                resize = (j == 0) and (i != 0)
                stage.add_module("unit{}".format(j + 1), VoVUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    branch_channels=branch_channels[i][j],
                    num_branches=num_branches,
                    resize=resize,
                    use_residual=use_residual))
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
        for module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_vovnet(blocks,
               slim=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    slim : bool, default False
        Whether to use a slim model.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if blocks == 27:
        layers = [1, 1, 1, 1]
    elif blocks == 39:
        layers = [1, 1, 2, 2]
    elif blocks == 57:
        layers = [1, 1, 4, 3]
    else:
        raise ValueError("Unsupported VoVNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 6 + 3 == blocks)

    num_branches = 5
    channels_per_layers = [256, 512, 768, 1024]
    branch_channels_per_layers = [128, 160, 192, 224]
    if slim:
        channels_per_layers = [ci // 2 for ci in channels_per_layers]
        branch_channels_per_layers = [ci // 2 for ci in branch_channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    branch_channels = [[ci] * li for (ci, li) in zip(branch_channels_per_layers, layers)]

    net = VoVNet(
        channels=channels,
        branch_channels=branch_channels,
        num_branches=num_branches,
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


def vovnet27s(**kwargs):
    """
    VoVNet-27-slim model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=27, slim=True, model_name="vovnet27s", **kwargs)


def vovnet39(**kwargs):
    """
    VoVNet-39 model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=39, model_name="vovnet39", **kwargs)


def vovnet57(**kwargs):
    """
    VoVNet-57 model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=57, model_name="vovnet57", **kwargs)


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
        vovnet27s,
        vovnet39,
        vovnet57,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != vovnet27s or weight_count == 3525736)
        assert (model != vovnet39 or weight_count == 22600296)
        assert (model != vovnet57 or weight_count == 36640296)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
