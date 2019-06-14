"""
    FishNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.
"""

__all__ = ['FishNet', 'fishnet99', 'fishnet150', 'InterpolationBlock', 'ChannelSqueeze']

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .common import pre_conv1x1_block, pre_conv3x3_block, conv1x1, SesquialteralHourglass, Identity
from .preresnet import PreResActivation
from .senet import SEInitBlock


def channel_squeeze(x,
                    groups):
    """
    Channel squeeze operation.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, channels_per_group, groups, height, width).sum(dim=2)
    return x


class ChannelSqueeze(nn.Module):
    """
    Channel squeeze layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups):
        super(ChannelSqueeze, self).__init__()
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_squeeze(x, self.groups)


class InterpolationBlock(nn.Module):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : float
        Multiplier for spatial size.
    mode : str, default 'nearest'
        Algorithm used for upsampling.
    align_corners : bool, default None
        Whether to align the corner pixels of the input and output tensors
    """
    def __init__(self,
                 scale_factor,
                 mode="nearest",
                 align_corners=None):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            input=x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)


class PreSEAttBlock(nn.Module):
    """
    FishNet specific Squeeze-and-Excitation attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    reduction : int, default 16
        Squeeze reduction value.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=16):
        super(PreSEAttBlock, self).__init__()
        mid_cannels = out_channels // reduction

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_cannels,
            bias=True)
        self.conv2 = conv1x1(
            in_channels=mid_cannels,
            out_channels=out_channels,
            bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class FishBottleneck(nn.Module):
    """
    FishNet bottleneck block for residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation):
        super(FishBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=dilation,
            dilation=dilation)
        self.conv3 = pre_conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class FishBlock(nn.Module):
    """
    FishNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    squeeze : bool, default False
        Whether to use a channel squeeze operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 squeeze=False):
        super(FishBlock, self).__init__()
        self.squeeze = squeeze
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = FishBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation)
        if self.squeeze:
            assert (in_channels // 2 == out_channels)
            self.c_squeeze = ChannelSqueeze(
                channels=in_channels,
                groups=2)
        elif self.resize_identity:
            self.identity_conv = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x):
        if self.squeeze:
            identity = self.c_squeeze(x)
        elif self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class DownUnit(nn.Module):
    """
    FishNet down unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(DownUnit, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_channels in enumerate(out_channels_list):
            self.blocks.add_module("block{}".format(i + 1), FishBlock(
                in_channels=in_channels,
                out_channels=out_channels))
            in_channels = out_channels
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return x


class UpUnit(nn.Module):
    """
    FishNet up unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 dilation=1):
        super(UpUnit, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_channels in enumerate(out_channels_list):
            squeeze = (dilation > 1) and (i == 0)
            self.blocks.add_module("block{}".format(i + 1), FishBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=dilation,
                squeeze=squeeze))
            in_channels = out_channels
        self.upsample = InterpolationBlock(scale_factor=2)

    def forward(self, x):
        x = self.blocks(x)
        x = self.upsample(x)
        return x


class SkipUnit(nn.Module):
    """
    FishNet skip connection unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(SkipUnit, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_channels in enumerate(out_channels_list):
            self.blocks.add_module("block{}".format(i + 1), FishBlock(
                in_channels=in_channels,
                out_channels=out_channels))
            in_channels = out_channels

    def forward(self, x):
        x = self.blocks(x)
        return x


class SkipAttUnit(nn.Module):
    """
    FishNet skip connection unit with attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(SkipAttUnit, self).__init__()
        mid_channels1 = in_channels // 2
        mid_channels2 = 2 * in_channels

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels1)
        self.conv2 = pre_conv1x1_block(
            in_channels=mid_channels1,
            out_channels=mid_channels2,
            bias=True)
        in_channels = mid_channels2

        self.se = PreSEAttBlock(
            in_channels=mid_channels2,
            out_channels=out_channels_list[-1])

        self.blocks = nn.Sequential()
        for i, out_channels in enumerate(out_channels_list):
            self.blocks.add_module("block{}".format(i + 1), FishBlock(
                in_channels=in_channels,
                out_channels=out_channels))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        w = self.se(x)
        x = self.blocks(x)
        x = x * w + w
        return x


class FishFinalBlock(nn.Module):
    """
    FishNet final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(FishFinalBlock, self).__init__()
        mid_channels = in_channels // 2

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.preactiv = PreResActivation(
            in_channels=mid_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.preactiv(x)
        return x


class FishNet(nn.Module):
    """
    FishNet model from 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.

    Parameters:
    ----------
    direct_channels : list of list of list of int
        Number of output channels for each unit along the straight path.
    skip_channels : list of list of list of int
        Number of output channels for each skip connection unit.
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
                 direct_channels,
                 skip_channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(FishNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        depth = len(direct_channels[0])
        down1_channels = direct_channels[0]
        up_channels = direct_channels[1]
        down2_channels = direct_channels[2]
        skip1_channels = skip_channels[0]
        skip2_channels = skip_channels[1]

        self.features = nn.Sequential()
        self.features.add_module("init_block", SEInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels

        down1_seq = nn.Sequential()
        skip1_seq = nn.Sequential()
        for i in range(depth + 1):
            skip1_channels_list = skip1_channels[i]
            if i < depth:
                skip1_seq.add_module("unit{}".format(i + 1), SkipUnit(
                    in_channels=in_channels,
                    out_channels_list=skip1_channels_list))
                down1_channels_list = down1_channels[i]
                down1_seq.add_module("unit{}".format(i + 1), DownUnit(
                    in_channels=in_channels,
                    out_channels_list=down1_channels_list))
                in_channels = down1_channels_list[-1]
            else:
                skip1_seq.add_module("unit{}".format(i + 1), SkipAttUnit(
                    in_channels=in_channels,
                    out_channels_list=skip1_channels_list))
                in_channels = skip1_channels_list[-1]

        up_seq = nn.Sequential()
        skip2_seq = nn.Sequential()
        for i in range(depth + 1):
            skip2_channels_list = skip2_channels[i]
            if i > 0:
                in_channels += skip1_channels[depth - i][-1]
            if i < depth:
                skip2_seq.add_module("unit{}".format(i + 1), SkipUnit(
                    in_channels=in_channels,
                    out_channels_list=skip2_channels_list))
                up_channels_list = up_channels[i]
                dilation = 2 ** i
                up_seq.add_module("unit{}".format(i + 1), UpUnit(
                    in_channels=in_channels,
                    out_channels_list=up_channels_list,
                    dilation=dilation))
                in_channels = up_channels_list[-1]
            else:
                skip2_seq.add_module("unit{}".format(i + 1), Identity())

        down2_seq = nn.Sequential()
        for i in range(depth):
            down2_channels_list = down2_channels[i]
            down2_seq.add_module("unit{}".format(i + 1), DownUnit(
                in_channels=in_channels,
                out_channels_list=down2_channels_list))
            in_channels = down2_channels_list[-1] + skip2_channels[depth - 1 - i][-1]

        self.features.add_module("hg", SesquialteralHourglass(
            down1_seq=down1_seq,
            skip1_seq=skip1_seq,
            up_seq=up_seq,
            skip2_seq=skip2_seq,
            down2_seq=down2_seq))
        self.features.add_module("final_block", FishFinalBlock(in_channels=in_channels))
        in_channels = in_channels // 2
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Sequential()
        self.output.add_module("final_conv", conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_fishnet(blocks,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create FishNet model with specific parameters.

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

    if blocks == 99:
        direct_layers = [[2, 2, 6], [1, 1, 1], [1, 2, 2]]
        skip_layers = [[1, 1, 1, 2], [4, 1, 1, 0]]
    elif blocks == 150:
        direct_layers = [[2, 4, 8], [2, 2, 2], [2, 2, 4]]
        skip_layers = [[2, 2, 2, 4], [4, 2, 2, 0]]
    else:
        raise ValueError("Unsupported FishNet with number of blocks: {}".format(blocks))

    direct_channels_per_layers = [[128, 256, 512], [512, 384, 256], [320, 832, 1600]]
    skip_channels_per_layers = [[64, 128, 256, 512], [512, 768, 512, 0]]

    direct_channels = [[[b] * c for (b, c) in zip(*a)] for a in
                       ([(ci, li) for (ci, li) in zip(direct_channels_per_layers, direct_layers)])]
    skip_channels = [[[b] * c for (b, c) in zip(*a)] for a in
                     ([(ci, li) for (ci, li) in zip(skip_channels_per_layers, skip_layers)])]

    init_block_channels = 64

    net = FishNet(
        direct_channels=direct_channels,
        skip_channels=skip_channels,
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


def fishnet99(**kwargs):
    """
    FishNet-99 model from 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_fishnet(blocks=99, model_name="fishnet99", **kwargs)


def fishnet150(**kwargs):
    """
    FishNet-150 model from 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_fishnet(blocks=150, model_name="fishnet150", **kwargs)


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
        fishnet99,
        fishnet150,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fishnet99 or weight_count == 16628904)
        assert (model != fishnet150 or weight_count == 24959400)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
