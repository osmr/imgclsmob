"""
    ChannelNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions,'
    https://arxiv.org/abs/1809.01330.
"""

__all__ = ['ChannelNet', 'channelnet']

import os
import torch
import torch.nn as nn
import torch.nn.init as init


def dwconv3x3(in_channels,
              out_channels,
              stride,
              bias=False):
    """
    3x3 depthwise version of the standard convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=out_channels,
        bias=bias)


class ChannetConv(nn.Module):
    """
    ChannelNet specific convolution block with Batch normalization and ReLU6 activation.

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
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Dropout rate.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 dropout_rate=0.0,
                 activate=True):
        super(ChannetConv, self).__init__()
        self.use_dropout = (dropout_rate > 0.0)
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            self.activ = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def channet_conv1x1(in_channels,
                    out_channels,
                    stride=1,
                    groups=1,
                    bias=False,
                    dropout_rate=0.0,
                    activate=True):
    """
    1x1 version of ChannelNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Dropout rate.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ChannetConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
        dropout_rate=dropout_rate,
        activate=activate)


def channet_conv3x3(in_channels,
                    out_channels,
                    stride,
                    padding=1,
                    dilation=1,
                    groups=1,
                    bias=False,
                    dropout_rate=0.0,
                    activate=True):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Dropout rate.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ChannetConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        dropout_rate=dropout_rate,
        activate=activate)


class ChannetDwsConvBlock(nn.Module):
    """
    ChannelNet specific depthwise separable convolution block with BatchNorms and activations at last convolution
    layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    dropout_rate : float, default 0.0
        Dropout rate.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 dropout_rate=0.0):
        super(ChannetDwsConvBlock, self).__init__()
        self.dw_conv = dwconv3x3(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=stride)
        self.pw_conv = channet_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class SimpleGroupBlock(nn.Module):
    """
    ChannelNet specific block with a sequence of depthwise separable group convolution layers.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    multi_blocks : int
        Number of DWS layers in the sequence.
    groups : int
        Number of groups.
    dropout_rate : float
        Dropout rate.
    """
    def __init__(self,
                 channels,
                 multi_blocks,
                 groups,
                 dropout_rate):
        super(SimpleGroupBlock, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(multi_blocks):
            self.blocks.add_module("block{}".format(i + 1), ChannetDwsConvBlock(
                in_channels=channels,
                out_channels=channels,
                stride=1,
                groups=groups,
                dropout_rate=dropout_rate))

    def forward(self, x):
        x = self.blocks(x)
        return x


class ChannelwiseConv2d(nn.Module):
    """
    ChannelNet specific block with channel-wise convolution.

    Parameters:
    ----------
    groups : int
        Number of groups.
    dropout_rate : float
        Dropout rate.
    """
    def __init__(self,
                 groups,
                 dropout_rate):
        super(ChannelwiseConv2d, self).__init__()
        self.use_dropout = (dropout_rate > 0.0)

        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=groups,
            kernel_size=(4 * groups, 1, 1),
            stride=(groups, 1, 1),
            padding=(2 * groups - 1, 0, 0),
            bias=False)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch, channels, height, width = x.size()
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = x.view(batch, channels, height, width)
        return x


class ConvGroupBlock(nn.Module):
    """
    ChannelNet specific block with a combination of channel-wise convolution, depthwise separable group convolutions.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    multi_blocks : int
        Number of DWS layers in the sequence.
    groups : int
        Number of groups.
    dropout_rate : float
        Dropout rate.
    """
    def __init__(self,
                 channels,
                 multi_blocks,
                 groups,
                 dropout_rate):
        super(ConvGroupBlock, self).__init__()
        self.conv = ChannelwiseConv2d(
            groups=groups,
            dropout_rate=dropout_rate)
        self.block = SimpleGroupBlock(
            channels=channels,
            multi_blocks=multi_blocks,
            groups=groups,
            dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        return x


class ChannetUnit(nn.Module):
    """
    ChannelNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : tuple/list of 2 int
        Number of output channels for each sub-block.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    multi_blocks : int
        Number of DWS layers in the sequence.
    groups : int
        Number of groups.
    dropout_rate : float
        Dropout rate.
    block_names : tuple/list of 2 str
        Sub-block names.
    merge_type : str
        Type of sub-block output merging.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 strides,
                 multi_blocks,
                 groups,
                 dropout_rate,
                 block_names,
                 merge_type):
        super(ChannetUnit, self).__init__()
        assert (len(block_names) == 2)
        assert (merge_type in ["seq", "add", "cat"])
        self.merge_type = merge_type

        self.blocks = nn.Sequential()
        for i, (out_channels, block_name) in enumerate(zip(out_channels_list, block_names)):
            stride_i = (strides if i == 0 else 1)
            if block_name == "channet_conv3x3":
                self.blocks.add_module("block{}".format(i + 1), channet_conv3x3(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride_i,
                    dropout_rate=dropout_rate,
                    activate=False))
            elif block_name == "channet_dws_conv_block":
                self.blocks.add_module("block{}".format(i + 1), ChannetDwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride_i,
                    dropout_rate=dropout_rate))
            elif block_name == "simple_group_block":
                self.blocks.add_module("block{}".format(i + 1), SimpleGroupBlock(
                    channels=in_channels,
                    multi_blocks=multi_blocks,
                    groups=groups,
                    dropout_rate=dropout_rate))
            elif block_name == "conv_group_block":
                self.blocks.add_module("block{}".format(i + 1), ConvGroupBlock(
                    channels=in_channels,
                    multi_blocks=multi_blocks,
                    groups=groups,
                    dropout_rate=dropout_rate))
            else:
                raise NotImplementedError()
            in_channels = out_channels

    def forward(self, x):
        x_outs = []
        for block in self.blocks._modules.values():
            x = block(x)
            x_outs.append(x)
        if self.merge_type == "add":
            for i in range(len(x_outs) - 1):
                x = x + x_outs[i]
        elif self.merge_type == "cat":
            x = torch.cat(tuple(x_outs), dim=1)
        return x


class ChannelNet(nn.Module):
    """
    ChannelNet model from 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise
    Convolutions,' https://arxiv.org/abs/1809.01330.

    Parameters:
    ----------
    channels : list of list of list of int
        Number of output channels for each unit.
    block_names : list of list of list of str
        Names of blocks for each unit.
    block_names : list of list of str
        Merge types for each unit.
    dropout_rate : float, default 0.0001
        Dropout rate.
    multi_blocks : int, default 2
        Block count architectural parameter.
    groups : int, default 2
        Group count architectural parameter.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 block_names,
                 merge_types,
                 dropout_rate=0.0001,
                 multi_blocks=2,
                 groups=2,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ChannelNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) else 1
                stage.add_module("unit{}".format(j + 1), ChannetUnit(
                    in_channels=in_channels,
                    out_channels_list=out_channels,
                    strides=strides,
                    multi_blocks=multi_blocks,
                    groups=groups,
                    dropout_rate=dropout_rate,
                    block_names=block_names[i][j],
                    merge_type=merge_types[i][j]))
                if merge_types[i][j] == "cat":
                    in_channels = sum(out_channels)
                else:
                    in_channels = out_channels[-1]
            self.features.add_module("stage{}".format(i + 1), stage)
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


def get_channelnet(model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".torch", "models"),
                   **kwargs):
    """
    Create ChannelNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    channels = [[[32, 64]], [[128, 128]], [[256, 256]], [[512, 512], [512, 512]], [[1024, 1024]]]
    block_names = [[["channet_conv3x3", "channet_dws_conv_block"]],
                   [["channet_dws_conv_block", "channet_dws_conv_block"]],
                   [["channet_dws_conv_block", "channet_dws_conv_block"]],
                   [["channet_dws_conv_block", "simple_group_block"], ["conv_group_block", "conv_group_block"]],
                   [["channet_dws_conv_block", "channet_dws_conv_block"]]]
    merge_types = [["cat"], ["cat"], ["cat"], ["add", "add"], ["seq"]]

    net = ChannelNet(
        channels=channels,
        block_names=block_names,
        merge_types=merge_types,
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


def channelnet(**kwargs):
    """
    ChannelNet model from 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise
    Convolutions,' https://arxiv.org/abs/1809.01330.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_channelnet(model_name="channelnet", **kwargs)


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
        channelnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != channelnet or weight_count == 3875112)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
