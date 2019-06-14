"""
    IGCV3 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.
"""

__all__ = ['IGCV3', 'igcv3_w1', 'igcv3_w3d4', 'igcv3_wd2', 'igcv3_wd4']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, ChannelShuffle


class InvResUnit(nn.Module):
    """
    So-called 'Inverted Residual Unit' layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    expansion : bool
        Whether do expansion of channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion):
        super(InvResUnit, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels
        groups = 2

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=groups,
            activation=None)
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=groups)
        self.conv2 = dwconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activation="relu6")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            groups=groups,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.c_shuffle(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class IGCV3(nn.Module):
    """
    IGCV3 model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
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
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(IGCV3, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            activation="relu6"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                stage.add_module("unit{}".format(j + 1), InvResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            activation="relu6"))
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


def get_igcv3(width_scale,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create IGCV3-D model with specific parameters.

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

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 4, 6, 8, 6, 6, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [[]])

    if width_scale != 1.0:
        def make_even(x):
            return x if (x % 2 == 0) else x + 1
        channels = [[make_even(int(cij * width_scale)) for cij in ci] for ci in channels]
        init_block_channels = make_even(int(init_block_channels * width_scale))
        if width_scale > 1.0:
            final_block_channels = make_even(int(final_block_channels * width_scale))

    net = IGCV3(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
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


def igcv3_w1(**kwargs):
    """
    IGCV3-D 1.0x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=1.0, model_name="igcv3_w1", **kwargs)


def igcv3_w3d4(**kwargs):
    """
    IGCV3-D 0.75x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.75, model_name="igcv3_w3d4", **kwargs)


def igcv3_wd2(**kwargs):
    """
    IGCV3-D 0.5x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.5, model_name="igcv3_wd2", **kwargs)


def igcv3_wd4(**kwargs):
    """
    IGCV3-D 0.25x model from 'IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks,'
    https://arxiv.org/abs/1806.00178.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_igcv3(width_scale=0.25, model_name="igcv3_wd4", **kwargs)


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
        igcv3_w1,
        igcv3_w3d4,
        igcv3_wd2,
        igcv3_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != igcv3_w1 or weight_count == 3491688)
        assert (model != igcv3_w3d4 or weight_count == 2638084)
        assert (model != igcv3_wd2 or weight_count == 1985528)
        assert (model != igcv3_wd4 or weight_count == 1534020)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
