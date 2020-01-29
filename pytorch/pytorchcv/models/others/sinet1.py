"""
    SINet for image segmentation, implemented in PyTorch.
    Original paper: 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.
"""

__all__ = ['SBNet_aux', 'sinet']

import numpy as np
from inspect import isfunction
import torch
import torch.nn as nn


class conv1x1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            groups=groups,
            bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.

    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.

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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
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
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.act = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.act(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
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
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 version of the standard convolution block.

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
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def round_channels(channels,
                   divisor=8):
    """
    Round weighted channel number (make divisible operation).

    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    activation : function, or str, or nn.Module, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Module, default 'sigmoid'
        Activation function after the last convolution.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 round_mid=False,
                 mid_activation=(lambda: nn.ReLU(inplace=True)),
                 out_activation=(lambda: nn.Sigmoid())):
        super(SEBlock, self).__init__()
        self.use_conv2 = reduction > 1
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            bias=True)
        if self.use_conv2:
            self.activ = get_activation_layer(mid_activation)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels,
                bias=True)
        self.sigmoid = get_activation_layer(out_activation)

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        if self.use_conv2:
            w = self.activ(w)
            w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


def dwconv_block(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
    """
    Depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class DwsConvBlock(nn.Module):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    reduction : int, default 0
        Squeeze reduction value (0 means no-se).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_eps=1e-5,
                 dw_activation=(lambda: nn.ReLU(inplace=True)),
                 pw_activation=(lambda: nn.ReLU(inplace=True)),
                 se_reduction=0):
        super(DwsConvBlock, self).__init__()
        self.use_se = se_reduction > 0

        self.dw_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            use_bn=dw_use_bn,
            bn_eps=bn_eps,
            activation=dw_activation)
        if self.use_se:
            self.se = SEBlock(
                channels=in_channels,
                reduction=se_reduction,
                round_mid=False,
                mid_activation=(lambda: nn.PReLU(in_channels // se_reduction)),
                out_activation=(lambda: nn.PReLU(in_channels)))
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=pw_use_bn,
            bn_eps=bn_eps,
            activation=pw_activation)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     stride=1,
                     padding=1,
                     dilation=1,
                     bias=False,
                     dw_use_bn=True,
                     pw_use_bn=True,
                     bn_eps=1e-5,
                     dw_activation=(lambda: nn.ReLU(inplace=True)),
                     pw_activation=(lambda: nn.ReLU(inplace=True)),
                     se_reduction=0):
    """
    3x3 depthwise separable version of the standard convolution block.

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
    bias : bool, default False
        Whether the layer uses a bias vector.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    reduction : int, default 0
        Squeeze reduction value (0 means no-se).
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        dw_use_bn=dw_use_bn,
        pw_use_bn=pw_use_bn,
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        se_reduction=se_reduction)


def dwconv3x3_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=1,
                    dilation=1,
                    bias=False,
                    bn_eps=1e-5,
                    activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 depthwise version of the standard convolution block.

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
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation)


class FDWConvBlock(nn.Module):
    """
    Factorized depthwise separable convolution block with BatchNorms and activations at each convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the each convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(FDWConvBlock, self).__init__()
        self.activate = (activation is not None)

        self.v_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=(padding, 0),
            dilation=dilation,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=None)
        self.h_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, padding),
            dilation=dilation,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=None)
        if self.activate:
            self.act = get_activation_layer(activation)

    def forward(self, x):
        x = self.v_conv(x) + self.h_conv(x)
        if self.activate:
            x = self.act(x)
        return x


def fdwconv3x3_block(in_channels,
                     out_channels,
                     stride=1,
                     padding=1,
                     dilation=1,
                     bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 factorized depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return FDWConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def fdwconv5x5_block(in_channels,
                     out_channels,
                     stride=1,
                     padding=2,
                     dilation=1,
                     bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation=(lambda: nn.ReLU(inplace=True))):
    """
    5x5 factorized depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return FDWConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class SBBlock(nn.Module):
    """
    SB-block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size for a factorized depthwise separable convolution block.
    scale_factor : int
        Scale factor.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor,
                 bn_eps=1e-5):
        super(SBBlock, self).__init__()
        self.use_scale = (scale_factor > 1)

        if self.use_scale:
            self.down_scale = nn.AvgPool2d(
                kernel_size=scale_factor,
                stride=scale_factor)
            self.up_scale = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        use_fdw = scale_factor > 0
        if use_fdw:
            fdwconv3x3_class = fdwconv3x3_block if kernel_size == 3 else fdwconv5x5_block
            self.conv1 = fdwconv3x3_class(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(in_channels)))
        else:
            self.conv1 = dwconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(in_channels)))

        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=bn_eps)

    def forward(self, x):
        if self.use_scale:
            x = self.down_scale(x)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_scale:
            x = self.up_scale(x)

        x = self.bn(x)
        return x


class PreActivation(nn.Module):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 bn_eps=1e-5):
        super(PreActivation, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features=in_channels,
            eps=bn_eps)
        self.activ = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class Concurrent(nn.Sequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    """
    def __init__(self,
                 axis=1,
                 stack=False):
        super(Concurrent, self).__init__()
        self.axis = axis
        self.stack = stack

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.stack:
            out = torch.stack(tuple(out), dim=self.axis)
        else:
            out = torch.cat(tuple(out), dim=self.axis)
        return out


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

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
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

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
        super(ChannelShuffle, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)


class ESPBlock(nn.Module):
    """
    ESP block, which is based on the following principle: Reduce ---> Split ---> Transform --> Merge.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_sizes : list of int
        Convolution window size for branches.
    scale_factors : list of int
        Scale factor for branches.
    use_residual : bool
        Whether to use residual connection.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 scale_factors,
                 use_residual,
                 bn_eps=1e-5):
        super(ESPBlock, self).__init__()
        self.use_residual = use_residual
        groups = len(kernel_sizes)

        mid_channels = int(out_channels / groups)
        res_channels = out_channels - groups * mid_channels

        self.conv = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=groups)

        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=groups)

        self.branches = Concurrent()
        for i in range(groups):
            out_channels_i = (mid_channels + res_channels) if i == 0 else mid_channels
            self.branches.add_module("branch{}".format(i + 2), SBBlock(
                    in_channels=mid_channels,
                    out_channels=out_channels_i,
                    kernel_size=kernel_sizes[i],
                    scale_factor=scale_factors[i],
                    bn_eps=bn_eps))

        self.preactiv = PreActivation(
            in_channels=out_channels,
            bn_eps=bn_eps)

    def forward(self, x):
        if self.use_residual:
            identity = x

        x = self.conv(x)
        x = self.c_shuffle(x)
        x = self.branches(x)

        if self.use_residual:
            x = identity + x

        x = self.preactiv(x)
        return x


class SBEncoder(nn.Module):
    def __init__(self,
                 kernel_sizes_list,
                 scale_factors_list,
                 classes=20,
                 p=5,
                 q=3,
                 chnn=1.0,
                 bn_eps=1e-5):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super(SBEncoder, self).__init__()
        dim1 = 24
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 72 + 4 * (chnn - 1)
        dim4 = 96 + 4 * (chnn - 1)

        bn_eps = 1e-3

        in_channels = 3
        out_channels = 16
        self.level1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

        in_channels = out_channels
        self.level2_0 = dwsconv3x3_block(
            in_channels=in_channels,
            out_channels=classes,
            stride=2,
            dw_use_bn=False,
            bn_eps=bn_eps,
            dw_activation=None,
            pw_activation=(lambda: nn.PReLU(classes)),
            se_reduction=1)

        self.level3_0 = dwsconv3x3_block(
            in_channels=classes,
            out_channels=dim1,
            stride=2,
            dw_use_bn=False,
            bn_eps=bn_eps,
            dw_activation=None,
            pw_activation=(lambda: nn.PReLU(dim1)),
            se_reduction=1)
        self.level3 = nn.ModuleList()
        in_channels = dim1
        out_channels = dim2
        for i in range(p):
            use_residual = (i != 0)
            kernel_sizes = kernel_sizes_list[i]
            scale_factors = scale_factors_list[i]
            self.level3.append(ESPBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                scale_factors=scale_factors,
                use_residual=use_residual,
                bn_eps=bn_eps))
            in_channels = out_channels
        self.BR3 = PreActivation(
            in_channels=(dim2 + dim1),
            bn_eps=bn_eps)

        self.level4_0 = dwsconv3x3_block(
            in_channels=(dim2 + dim1),
            out_channels=dim2,
            stride=2,
            dw_use_bn=False,
            bn_eps=bn_eps,
            dw_activation=None,
            pw_activation=(lambda: nn.PReLU(dim2)),
            se_reduction=2)
        self.level4 = nn.ModuleList()
        out_channels = dim3
        for i in range(q // 2):
            use_residual = (i != 0)
            kernel_sizes = kernel_sizes_list[p + i]
            scale_factors = scale_factors_list[p + i]
            self.level4.append(ESPBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                scale_factors=scale_factors,
                use_residual=use_residual,
                bn_eps=bn_eps))
            in_channels = out_channels
        out_channels = dim4
        for i in range(q // 2, q):
            kernel_sizes = kernel_sizes_list[p + i]
            scale_factors = scale_factors_list[p + i]
            use_residual = (i != q // 2)
            self.level4.append(ESPBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                scale_factors=scale_factors,
                use_residual=use_residual,
                bn_eps=bn_eps))
            in_channels = out_channels
        self.BR4 = PreActivation(
            in_channels=(dim4 + dim2),
            bn_eps=bn_eps)

        self.classifier = conv1x1(
            in_channels=(dim4 + dim2),
            out_channels=classes)

    def forward(self, input):
        output1 = self.level1(input) #8h 8w
        output2_0 = self.level2_0(output1)  # 4h 4w
        output3_0 = self.level3_0(output2_0)  # 4h 4w

        # print(str(output1_0.size()))
        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3) # 2h 2w

        output4_0 = self.level4_0(self.BR3(torch.cat([output3_0, output3],1)))  # h w
        # print(str(output2_0.size()))

        for i, layer in enumerate(self.level4):
            if i == 0:
                output4 = layer(output4_0)
            else:
                output4 = layer(output4)

        output4_cat = self.BR4(torch.cat([output4_0, output4], 1))

        classifier = self.classifier(output4_cat)

        return classifier


class SBNet_aux(nn.Module):

    def __init__(self,
                 kernel_sizes_list,
                 scale_factors_list,
                 classes=20,
                 p=2,
                 q=3,
                 chnn=1.0,
                 bn_eps=1e-5):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        # print("SB Net Enc bracnch num :  " + str(len(config[0])))
        # print("SB Net Enc chnn num:  " + str(chnn))
        # dim1 = 24
        dim2 = 48 + 4 * (chnn - 1)
        # dim3 = 72 + 4 * (chnn - 1)
        # dim4 = 96 + 4 * (chnn - 1)

        self.encoder = SBEncoder(
            kernel_sizes_list,
            scale_factors_list,
            classes,
            p,
            q,
            chnn)

        self.up = nn.functional.interpolate
        self.bn_4 = nn.BatchNorm2d(classes, eps=1e-03)

        self.level3_C = conv1x1_block(
            in_channels=dim2,
            out_channels=classes,
            bn_eps=1e-3,
            activation=(lambda: nn.PReLU(classes)))
        self.bn_3 = nn.BatchNorm2d(classes, eps=1e-03)

        self.classifier = nn.ConvTranspose2d(
            classes,
            classes,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False)

    def forward(self, input, train=False):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output1 = self.encoder.level1(input)  # 8h 8w
        output2_0 = self.encoder.level2_0(output1)  # 4h 4w
        output3_0 = self.encoder.level3_0(output2_0)  # 2h 2w

        # print(str(output1_0.size()))
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)  # 2h 2w

        output4_0 = self.encoder.level4_0(self.encoder.BR3(torch.cat([output3_0, output3], 1)))  # h w
        # print(str(output2_0.size()))

        for i, layer in enumerate(self.encoder.level4):
            if i == 0:
                output4 = layer(output4_0)
            else:
                output4 = layer(output4)

        output4_cat = self.encoder.BR4(torch.cat([output4_0, output4], 1))
        Enc_final = self.encoder.classifier(output4_cat)

        Dnc_stage1 = self.bn_4(self.up(Enc_final, scale_factor=2, mode="bilinear"))  # 2h 2w
        stage1_confidence = nn.Softmax2d()(Dnc_stage1)
        b, c, h, w = Dnc_stage1.size()
        # Coarse_att = ((torch.max(Coarse_confidence,dim=1)[0]).unsqueeze(1)).expand(b,c,h,w)
        stage1_blocking = (torch.max(stage1_confidence, dim=1)[0]).unsqueeze(1).expand(b, c, h, w)

        Dnc_stage2_0 = self.level3_C(output3)  # 2h 2w
        Dnc_stage2 = self.bn_3(
            self.up(Dnc_stage2_0 * (1 - stage1_blocking) + (Dnc_stage1), scale_factor=2, mode="bilinear"))  # 4h 4w

        stage2_confidence = nn.Softmax2d()(Dnc_stage2)  # 4h 4w
        b, c, h, w = Dnc_stage2.size()

        stage2_blocking = (torch.max(stage2_confidence, dim=1)[0]).unsqueeze(1).expand(b, c, h, w)
        Dnc_stage3 = output2_0 * (1 - stage2_blocking) + (Dnc_stage2)

        classifier = self.classifier(Dnc_stage3)

        if train:
            return Enc_final, classifier
        else:
            return classifier


def sinet(classes=19,
          p=3,
          q=10,
          chnn=4,
          encoderFile=None,
          pretrained=False):
    kernel_sizes_list =\
        [[3, 5], [3, 3], [3, 3], [3, 5], [3, 3], [5, 5], [3, 5], [3, 5], [3, 5], [3, 3], [5, 5], [3, 5], [3, 5]]
    scale_factors_list =\
        [[1, 1], [0, 1], [0, 1], [1, 1], [0, 1], [1, 4], [2, 8], [1, 1], [1, 1], [0, 1], [1, 8], [2, 4], [0, 2]]
    model = SBNet_aux(
        kernel_sizes_list=kernel_sizes_list,
        scale_factors_list=scale_factors_list,
        classes=classes,
        p=p,
        q=q,
        chnn=chnn)
    return model


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
        sinet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sinet or weight_count == 119418)

        x = torch.randn(14, 3, 1024, 2048)
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (14, 19, 512, 1024))


if __name__ == "__main__":
    _test()
