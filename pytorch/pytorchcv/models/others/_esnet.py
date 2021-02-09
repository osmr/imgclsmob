"""
    ESNet for image segmentation, implemented in PyTorch.
    Original paper: 'ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1906.09826.
"""

__all__ = ['ESNet']

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import conv3x3, ConvBlock, NormActivation, DeconvBlock, Concurrent


def deconv3x3_block(padding=1,
                    out_padding=1,
                    **kwargs):
    """
    3x3 version of the deconvolution block with batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the deconvolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for deconvolution layer.
    ext_padding : tuple/list of 4 int, default None
        Extra padding value for deconvolution layer.
    out_padding : int or tuple/list of 2 int, default 1
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return DeconvBlock(
        kernel_size=3,
        padding=padding,
        out_padding=out_padding,
        **kwargs)


class AsymConvBlock(nn.Module):
    """
    Asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    lw_use_bn : bool, default True
        Whether to use BatchNorm layer (leftwise convolution block).
    rw_use_bn : bool, default True
        Whether to use BatchNorm layer (rightwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    lw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the leftwise convolution block.
    rw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the rightwise convolution block.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 lw_use_bn=True,
                 rw_use_bn=True,
                 bn_eps=1e-5,
                 lw_activation=(lambda: nn.ReLU(inplace=True)),
                 rw_activation=(lambda: nn.ReLU(inplace=True))):
        super(AsymConvBlock, self).__init__()
        self.lw_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=groups,
            bias=bias,
            use_bn=lw_use_bn,
            bn_eps=bn_eps,
            activation=lw_activation)
        self.rw_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            bias=bias,
            use_bn=rw_use_bn,
            bn_eps=bn_eps,
            activation=rw_activation)

    def forward(self, x):
        x = self.lw_conv(x)
        x = self.rw_conv(x)
        return x


class LEDDownBlock(nn.Module):
    """
    LEDNet specific downscale block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 correct_size_mismatch,
                 bn_eps):
        super(LEDDownBlock, self).__init__()
        self.correct_size_mismatch = correct_size_mismatch

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=(out_channels - in_channels),
            stride=2,
            bias=True)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps)

    def forward(self, x):
        y1 = self.pool(x)
        y2 = self.conv(x)

        if self.correct_size_mismatch:
            diff_h = y2.size()[2] - y1.size()[2]
            diff_w = y2.size()[3] - y1.size()[3]
            y1 = F.pad(y1, pad=(diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))

        x = torch.cat((y2, y1), dim=1)
        x = self.norm_activ(x)
        return x


class FCU(nn.Module):
    """
    Factorized convolution unit.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(FCU, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps)
        self.conv2 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)

        x = x + identity
        x = self.activ(x)
        return x


class PFCUBranch(nn.Module):
    """
    Parallel factorized convolution unit's branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(PFCUBranch, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=dilation,
            dilation=dilation,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class PFCU(nn.Module):
    """
    Parallel factorized convolution unit.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 dropout_rate,
                 bn_eps):
        super(PFCU, self).__init__()
        dilations = [2, 5, 9]
        padding = (kernel_size - 1) // 2

        self.conv1 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps)
        self.branches = Concurrent(merge_type="sum")
        for i, dilation in enumerate(dilations):
            self.branches.add_module("branch{}".format(i + 1), PFCUBranch(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bn_eps=bn_eps))
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.branches(x)

        x = x + identity
        x = self.activ(x)
        return x


class ESNet(nn.Module):
    def __init__(self,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 in_channels=3,
                 num_classes=19):
        super().__init__()

        layers = [[4, 3, 4], [3, 3]]
        channels = [[16, 64, 128], [64, 16]]
        kernel_sizes = [[3, 5, 3], [5, 3]]
        dropout_rates = [[0.03, 0.03, 0.3], [0, 0]]

        self.encoder = nn.Sequential()
        for i, layers_per_stage in enumerate(layers[0]):
            out_channels = channels[0][i]
            kernel_size = kernel_sizes[0][i]
            dropout_rate = dropout_rates[0][i]
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), LEDDownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        correct_size_mismatch=correct_size_mismatch,
                        bn_eps=bn_eps))
                    in_channels = out_channels
                elif i != len(layers[0]) - 1:
                    stage.add_module("unit{}".format(j + 1), FCU(
                        channels=in_channels,
                        kernel_size=kernel_size,
                        dilation=1,
                        dropout_rate=dropout_rate,
                        bn_eps=bn_eps))
                else:
                    stage.add_module("unit{}".format(j + 1), PFCU(
                        channels=in_channels,
                        kernel_size=kernel_size,
                        dropout_rate=dropout_rate,
                        bn_eps=bn_eps))
            self.encoder.add_module("stage{}".format(i + 1), stage)

        self.decoder = nn.Sequential()
        for i, layers_per_stage in enumerate(layers[1]):
            out_channels = channels[1][i]
            kernel_size = kernel_sizes[1][i]
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), deconv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        bias=True,
                        bn_eps=bn_eps))
                    in_channels = out_channels
                else:
                    stage.add_module("unit{}".format(j + 1), FCU(
                        channels=in_channels,
                        kernel_size=kernel_size,
                        dilation=1,
                        dropout_rate=0,
                        bn_eps=bn_eps))
            self.decoder.add_module("stage{}".format(i + 1), stage)

        self.head = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x


def esnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    bn_eps = 1e-3

    return ESNet(num_classes=num_classes, bn_eps=bn_eps, **kwargs)


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
        esnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != esnet_cityscapes or weight_count == 1660607)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
