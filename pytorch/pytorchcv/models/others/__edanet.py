"""
    EDANet for image segmentation, implemented in PyTorch.
    Original paper: 'Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1809.06323.
    : https://github.com/shaoyuanlo/EDANet
"""

import torch
import torch.nn as nn
from common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, ConvBlock, NormActivation, Concurrent,\
    InterpolationBlock, DualPathSequential
from lednet import asym_conv3x3_block


__all__ = ["EDANet"]


class DownBlock(nn.Module):
    """
    EDANet specific downsample block for the main branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps):
        super(DownBlock, self).__init__()
        self.expand = (in_channels < out_channels)
        mid_channels = out_channels - in_channels if self.expand else out_channels

        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            stride=2)
        if self.expand:
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps)

    def forward(self, x):
        y = self.conv(x)

        if self.expand:
            z = self.pool(x)
            y = torch.cat((y, z), dim=1)

        y = self.norm_activ(y)
        return y


class EDABlock(nn.Module):
    """
    EDANet base block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(EDABlock, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = asym_conv3x3_block(
            channels=channels,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            lw_activation=None)
        self.conv2 = asym_conv3x3_block(
            channels=channels,
            padding=dilation,
            dilation=dilation,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class EDAUnit(nn.Module):
    """
    EDANet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(EDAUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        mid_channels = out_channels - in_channels

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True)
        self.conv2 = EDABlock(
            channels=mid_channels,
            dilation=dilation,
            dropout_rate=dropout_rate,
            bn_eps=bn_eps)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.cat((x, identity), dim=1)
        x = self.activ(x)
        return x


class EDANet(nn.Module):
    def __init__(self,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(EDANet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        growth_rate = 40
        channels = [15, 60, 130, 450]
        dilations = [[0], [0, 1, 1, 1, 2, 2], [0, 2, 2, 4, 4, 8, 8, 16, 16]]

        self.features = nn.Sequential()
        for i, dilations_per_stage in enumerate(dilations):
            out_channels = channels[i]
            stage = nn.Sequential()
            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), DownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bn_eps=bn_eps))
                else:
                    out_channels += growth_rate
                    stage.add_module("unit{}".format(j + 1), EDAUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dilation=dilation,
                        dropout_rate=0.02,
                        bn_eps=bn_eps))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.head = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=True)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = self.up(x)
        return x


def oth_edanet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    bn_eps = 1e-3
    return EDANet(num_classes=num_classes, bn_eps=bn_eps, **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    # fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        oth_edanet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_edanet_cityscapes or weight_count == 689485)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
