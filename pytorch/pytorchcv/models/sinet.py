"""
    SINet for image segmentation, implemented in PyTorch.
    Original paper: 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.
"""

__all__ = ['SINet', 'sinet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1, get_activation_layer, conv1x1_block, conv3x3_block, round_channels, dwconv_block,\
    Concurrent, InterpolationBlock, ChannelShuffle


class SEBlock(nn.Module):
    """
    SINet version of Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,'
    https://arxiv.org/abs/1709.01507.

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
        self.use_conv2 = (reduction > 1)
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(
            in_features=channels,
            out_features=mid_channels)
        if self.use_conv2:
            self.activ = get_activation_layer(mid_activation)
            self.fc2 = nn.Linear(
                in_features=mid_channels,
                out_features=channels)
        self.sigmoid = get_activation_layer(out_activation)

    def forward(self, x):
        w = self.pool(x)
        w = w.squeeze(dim=-1).squeeze(dim=-1)
        w = self.fc1(w)
        if self.use_conv2:
            w = self.activ(w)
            w = self.fc2(w)
        w = self.sigmoid(w)
        w = w.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x * w
        return x


class DwsConvBlock(nn.Module):
    """
    SINet version of depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    se_reduction : int, default 0
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
        self.use_se = (se_reduction > 0)

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
    3x3 depthwise separable version of the standard convolution block (SINet version).

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
    se_reduction : int, default 0
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
    3x3 depthwise version of the standard convolution block (SINet version).

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
        assert use_bn
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
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor,
                 bn_eps):
        super(SBBlock, self).__init__()
        self.use_scale = (scale_factor > 1)

        if self.use_scale:
            self.down_scale = nn.AvgPool2d(
                kernel_size=scale_factor,
                stride=scale_factor)
            self.up_scale = InterpolationBlock(scale_factor=scale_factor)

        use_fdw = (scale_factor > 0)
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
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 bn_eps):
        super(PreActivation, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features=in_channels,
            eps=bn_eps)
        self.activ = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


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
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 scale_factors,
                 use_residual,
                 bn_eps):
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
            self.branches.add_module("branch{}".format(i + 1), SBBlock(
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


class SBStage(nn.Module):
    """
    SB stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    down_channels : int
        Number of output channels for a downscale block.
    channels_list : list of int
        Number of output channels for all residual block.
    kernel_sizes_list : list of int
        Convolution window size for branches.
    scale_factors_list : list of int
        Scale factor for branches.
    use_residual_list : list of int
        List of flags for using residual in each ESP-block.
    se_reduction : int
        Squeeze reduction value (0 means no-se).
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 down_channels,
                 channels_list,
                 kernel_sizes_list,
                 scale_factors_list,
                 use_residual_list,
                 se_reduction,
                 bn_eps):
        super(SBStage, self).__init__()
        self.down_conv = dwsconv3x3_block(
            in_channels=in_channels,
            out_channels=down_channels,
            stride=2,
            dw_use_bn=False,
            bn_eps=bn_eps,
            dw_activation=None,
            pw_activation=(lambda: nn.PReLU(down_channels)),
            se_reduction=se_reduction)
        in_channels = down_channels

        self.main_branch = nn.Sequential()
        for i, out_channels in enumerate(channels_list):
            use_residual = (use_residual_list[i] == 1)
            kernel_sizes = kernel_sizes_list[i]
            scale_factors = scale_factors_list[i]
            self.main_branch.add_module("block{}".format(i + 1), ESPBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                scale_factors=scale_factors,
                use_residual=use_residual,
                bn_eps=bn_eps))
            in_channels = out_channels

        self.preactiv = PreActivation(
            in_channels=(down_channels + in_channels),
            bn_eps=bn_eps)

    def forward(self, x):
        x = self.down_conv(x)
        y = self.main_branch(x)
        x = torch.cat((x, y), dim=1)
        x = self.preactiv(x)
        return x, y


class SBEncoderInitBlock(nn.Module):
    """
    SB encoder specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bn_eps):
        super(SBEncoderInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid_channels)))
        self.conv2 = dwsconv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=2,
            dw_use_bn=False,
            bn_eps=bn_eps,
            dw_activation=None,
            pw_activation=(lambda: nn.PReLU(out_channels)),
            se_reduction=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SBEncoder(nn.Module):
    """
    SB encoder for SINet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    init_block_channels : list int
        Number of output channels for convolutions in the initial block.
    down_channels_list : list of int
        Number of downsample channels for each residual block.
    channels_list : list of list of int
        Number of output channels for all residual block.
    kernel_sizes_list : list of list of int
        Convolution window size for each residual block.
    scale_factors_list : list of list of int
        Scale factor for each residual block.
    use_residual_list : list of list of int
        List of flags for using residual in each residual block.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 init_block_channels,
                 down_channels_list,
                 channels_list,
                 kernel_sizes_list,
                 scale_factors_list,
                 use_residual_list,
                 bn_eps):
        super(SBEncoder, self).__init__()
        self.init_block = SBEncoderInitBlock(
            in_channels=in_channels,
            mid_channels=init_block_channels[0],
            out_channels=init_block_channels[1],
            bn_eps=bn_eps)

        in_channels = init_block_channels[1]
        self.stage1 = SBStage(
            in_channels=in_channels,
            down_channels=down_channels_list[0],
            channels_list=channels_list[0],
            kernel_sizes_list=kernel_sizes_list[0],
            scale_factors_list=scale_factors_list[0],
            use_residual_list=use_residual_list[0],
            se_reduction=1,
            bn_eps=bn_eps)

        in_channels = down_channels_list[0] + channels_list[0][-1]
        self.stage2 = SBStage(
            in_channels=in_channels,
            down_channels=down_channels_list[1],
            channels_list=channels_list[1],
            kernel_sizes_list=kernel_sizes_list[1],
            scale_factors_list=scale_factors_list[1],
            use_residual_list=use_residual_list[1],
            se_reduction=2,
            bn_eps=bn_eps)

        in_channels = down_channels_list[1] + channels_list[1][-1]
        self.output = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        y1 = self.init_block(x)
        x, y2 = self.stage1(y1)
        x, _ = self.stage2(x)
        x = self.output(x)
        return x, y2, y1


class SBDecodeBlock(nn.Module):
    """
    SB decoder block for SINet.

    Parameters:
    ----------
    channels : int
        Number of output classes.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 bn_eps):
        super(SBDecodeBlock, self).__init__()
        self.up = InterpolationBlock(
            scale_factor=2,
            align_corners=False)
        self.bn = nn.BatchNorm2d(
            num_features=channels,
            eps=bn_eps)
        self.conf = nn.Softmax2d()

    def forward(self, x, y):
        x = self.up(x)
        x = self.bn(x)
        w_conf = self.conf(x)
        w_max = (torch.max(w_conf, dim=1)[0]).unsqueeze(1).expand_as(x)
        x = y * (1 - w_max) + x
        return x


class SBDecoder(nn.Module):
    """
    SB decoder for SINet.

    Parameters:
    ----------
    dim2 : int
        Size of dimension #2.
    num_classes : int
        Number of segmentation classes.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 dim2,
                 num_classes,
                 bn_eps):
        super(SBDecoder, self).__init__()
        self.decode1 = SBDecodeBlock(
            channels=num_classes,
            bn_eps=bn_eps)
        self.decode2 = SBDecodeBlock(
            channels=num_classes,
            bn_eps=bn_eps)
        self.conv3c = conv1x1_block(
            in_channels=dim2,
            out_channels=num_classes,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(num_classes)))
        self.output = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False)
        self.up = InterpolationBlock(scale_factor=2)

    def forward(self, y3, y2, y1):
        y2 = self.conv3c(y2)
        x = self.decode1(y3, y2)
        x = self.decode2(x, y1)
        x = self.output(x)
        x = self.up(x)
        return x


class SINet(nn.Module):
    """
    SINet model from 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.

    Parameters:
    ----------
    down_channels_list : list of int
        Number of downsample channels for each residual block.
    channels_list : list of list of int
        Number of output channels for all residual block.
    kernel_sizes_list : list of list of int
        Convolution window size for each residual block.
    scale_factors_list : list of list of int
        Scale factor for each residual block.
    use_residual_list : list of list of int
        List of flags for using residual in each residual block.
    dim2 : int
        Size of dimension #2.
    bn_eps : float
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    num_classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 down_channels_list,
                 channels_list,
                 kernel_sizes_list,
                 scale_factors_list,
                 use_residual_list,
                 dim2,
                 bn_eps,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=21):
        super(SINet, self).__init__()
        assert (fixed_size is not None)
        assert (in_channels > 0)
        assert ((in_size[0] % 64 == 0) and (in_size[1] % 64 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.aux = aux

        init_block_channels = [16, num_classes]
        out_channels = num_classes
        self.encoder = SBEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            init_block_channels=init_block_channels,
            down_channels_list=down_channels_list,
            channels_list=channels_list,
            kernel_sizes_list=kernel_sizes_list,
            scale_factors_list=scale_factors_list,
            use_residual_list=use_residual_list,
            bn_eps=bn_eps)

        self.decoder = SBDecoder(
            dim2=dim2,
            num_classes=num_classes,
            bn_eps=bn_eps)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        y3, y2, y1 = self.encoder(x)
        x = self.decoder(y3, y2, y1)
        if self.aux:
            return x, y3
        else:
            return x


def get_sinet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create SINet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    kernel_sizes_list = [
        [[3, 5], [3, 3], [3, 3]],
        [[3, 5], [3, 3], [5, 5], [3, 5], [3, 5], [3, 5], [3, 3], [5, 5], [3, 5], [3, 5]]]
    scale_factors_list = [
        [[1, 1], [0, 1], [0, 1]],
        [[1, 1], [0, 1], [1, 4], [2, 8], [1, 1], [1, 1], [0, 1], [1, 8], [2, 4], [0, 2]]]

    chnn = 4
    dims = [24] + [24 * (i + 2) + 4 * (chnn - 1) for i in range(3)]

    dim1 = dims[0]
    dim2 = dims[1]
    dim3 = dims[2]
    dim4 = dims[3]

    p = len(kernel_sizes_list[0])
    q = len(kernel_sizes_list[1])

    channels_list = [[dim2] * p, ([dim3] * (q // 2)) + ([dim4] * (q - q // 2))]
    use_residual_list = [[0] + ([1] * (p - 1)), [0] + ([1] * (q // 2 - 1)) + [0] + ([1] * (q - q // 2 - 1))]

    down_channels_list = [dim1, dim2]

    net = SINet(
        down_channels_list=down_channels_list,
        channels_list=channels_list,
        kernel_sizes_list=kernel_sizes_list,
        scale_factors_list=scale_factors_list,
        use_residual_list=use_residual_list,
        dim2=dims[1],
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


def sinet_cityscapes(num_classes=19, **kwargs):
    """
    SINet model for Cityscapes from 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze
    Modules and Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sinet(num_classes=num_classes, bn_eps=1e-3, model_name="sinet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    in_size = (1024, 2048)
    aux = False
    fixed_size = True
    pretrained = False

    models = [
        sinet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, aux=aux, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sinet_cityscapes or weight_count == 119418)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        ys = net(x)
        y = ys[0] if aux else ys
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
