"""
    SINet for image segmentation, implemented in Gluon.
    Original paper: 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.
"""

__all__ = ['SINet', 'sinet_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import PReLU2, conv1x1, get_activation_layer, conv1x1_block, conv3x3_block, round_channels, dwconv_block,\
    InterpolationBlock, ChannelShuffle


class SEBlock(HybridBlock):
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
                 mid_activation=(lambda: nn.Activation("relu")),
                 out_activation=(lambda: nn.Activation("sigmoid")),
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.use_conv2 = (reduction > 1)
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        with self.name_scope():
            self.fc1 = nn.Dense(
                in_units=channels,
                units=mid_channels)
            if self.use_conv2:
                self.activ = get_activation_layer(mid_activation)
                self.fc2 = nn.Dense(
                    in_units=mid_channels,
                    units=channels)
            self.sigmoid = get_activation_layer(out_activation)

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = F.Flatten(w)
        w = self.fc1(w)
        if self.use_conv2:
            w = self.activ(w)
            w = self.fc2(w)
        w = self.sigmoid(w)
        w = w.expand_dims(2).expand_dims(3).broadcast_like(x)
        x = x * w
        return x


class DwsConvBlock(HybridBlock):
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
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_epsilon : float, default 1e-5
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
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_epsilon=1e-5,
                 dw_activation=(lambda: nn.Activation("relu")),
                 pw_activation=(lambda: nn.Activation("relu")),
                 se_reduction=0,
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        self.use_se = (se_reduction > 0)

        with self.name_scope():
            self.dw_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                use_bias=use_bias,
                use_bn=dw_use_bn,
                bn_epsilon=bn_epsilon,
                activation=dw_activation)
            if self.use_se:
                self.se = SEBlock(
                    channels=in_channels,
                    reduction=se_reduction,
                    round_mid=False,
                    mid_activation=(lambda: PReLU2(in_channels // se_reduction)),
                    out_activation=(lambda: PReLU2(in_channels)))
            self.pw_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=pw_use_bn,
                bn_epsilon=bn_epsilon,
                activation=pw_activation)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=1,
                     dilation=1,
                     use_bias=False,
                     dw_use_bn=True,
                     pw_use_bn=True,
                     bn_epsilon=1e-5,
                     dw_activation=(lambda: nn.Activation("relu")),
                     pw_activation=(lambda: nn.Activation("relu")),
                     se_reduction=0,
                     **kwargs):
    """
    3x3 depthwise separable version of the standard convolution block (SINet version).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_epsilon : float, default 1e-5
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
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        dw_use_bn=dw_use_bn,
        pw_use_bn=pw_use_bn,
        bn_epsilon=bn_epsilon,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        se_reduction=se_reduction,
        **kwargs)


def dwconv3x3_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    bn_epsilon=1e-5,
                    activation=(lambda: nn.Activation("relu")),
                    **kwargs):
    """
    3x3 depthwise version of the standard convolution block (SINet version).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        activation=activation,
        **kwargs)


class FDWConvBlock(HybridBlock):
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
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the each convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(FDWConvBlock, self).__init__(**kwargs)
        assert use_bn
        self.activate = (activation is not None)

        with self.name_scope():
            self.v_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                strides=strides,
                padding=(padding, 0),
                dilation=dilation,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_epsilon=bn_epsilon,
                activation=None)
            self.h_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                strides=strides,
                padding=(0, padding),
                dilation=dilation,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_epsilon=bn_epsilon,
                activation=None)
            if self.activate:
                self.act = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.v_conv(x) + self.h_conv(x)
        if self.activate:
            x = self.act(x)
        return x


def fdwconv3x3_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=1,
                     dilation=1,
                     use_bias=False,
                     use_bn=True,
                     bn_epsilon=1e-5,
                     activation=(lambda: nn.Activation("relu")),
                     **kwargs):
    """
    3x3 factorized depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return FDWConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        activation=activation,
        **kwargs)


def fdwconv5x5_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=2,
                     dilation=1,
                     use_bias=False,
                     use_bn=True,
                     bn_epsilon=1e-5,
                     activation=(lambda: nn.Activation("relu")),
                     **kwargs):
    """
    5x5 factorized depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return FDWConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        activation=activation,
        **kwargs)


class SBBlock(HybridBlock):
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
    size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor,
                 size,
                 bn_epsilon,
                 **kwargs):
        super(SBBlock, self).__init__(**kwargs)
        self.use_scale = (scale_factor > 1)

        with self.name_scope():
            if self.use_scale:
                self.down_scale = nn.AvgPool2D(
                    pool_size=scale_factor,
                    strides=scale_factor)
                self.up_scale = InterpolationBlock(
                    scale_factor=scale_factor,
                    out_size=size)

            use_fdw = (scale_factor > 0)
            if use_fdw:
                fdwconv3x3_class = fdwconv3x3_block if kernel_size == 3 else fdwconv5x5_block
                self.conv1 = fdwconv3x3_class(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    bn_epsilon=bn_epsilon,
                    activation=(lambda: PReLU2(in_channels)))
            else:
                self.conv1 = dwconv3x3_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    bn_epsilon=bn_epsilon,
                    activation=(lambda: PReLU2(in_channels)))

            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                epsilon=bn_epsilon)

    def hybrid_forward(self, F, x):
        if self.use_scale:
            x = self.down_scale(x)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_scale:
            x = self.up_scale(x)

        x = self.bn(x)
        return x


class PreActivation(HybridBlock):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 bn_epsilon=1e-5,
                 **kwargs):
        super(PreActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                epsilon=bn_epsilon)
            self.activ = PReLU2(in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ESPBlock(HybridBlock):
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
    in_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 scale_factors,
                 use_residual,
                 in_size,
                 bn_epsilon,
                 **kwargs):
        super(ESPBlock, self).__init__(**kwargs)
        self.use_residual = use_residual
        groups = len(kernel_sizes)

        mid_channels = int(out_channels / groups)
        res_channels = out_channels - groups * mid_channels

        with self.name_scope():
            self.conv = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=groups)

            self.c_shuffle = ChannelShuffle(
                channels=mid_channels,
                groups=groups)

            self.branches = HybridConcurrent(axis=1, prefix="")
            with self.branches.name_scope():
                for i in range(groups):
                    out_channels_i = (mid_channels + res_channels) if i == 0 else mid_channels
                    self.branches.add(SBBlock(
                        in_channels=mid_channels,
                        out_channels=out_channels_i,
                        kernel_size=kernel_sizes[i],
                        scale_factor=scale_factors[i],
                        size=in_size,
                        bn_epsilon=bn_epsilon))

            self.preactiv = PreActivation(
                in_channels=out_channels,
                bn_epsilon=bn_epsilon)

    def hybrid_forward(self, F, x):
        if self.use_residual:
            identity = x

        x = self.conv(x)
        x = self.c_shuffle(x)
        x = self.branches(x)

        if self.use_residual:
            x = identity + x

        x = self.preactiv(x)
        return x


class SBStage(HybridBlock):
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
    in_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_epsilon : float
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
                 in_size,
                 bn_epsilon,
                 **kwargs):
        super(SBStage, self).__init__(**kwargs)
        with self.name_scope():
            self.down_conv = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=down_channels,
                strides=2,
                dw_use_bn=False,
                bn_epsilon=bn_epsilon,
                dw_activation=None,
                pw_activation=(lambda: PReLU2(down_channels)),
                se_reduction=se_reduction)
            in_channels = down_channels

            self.main_branch = nn.HybridSequential(prefix="")
            with self.main_branch.name_scope():
                for i, out_channels in enumerate(channels_list):
                    use_residual = (use_residual_list[i] == 1)
                    kernel_sizes = kernel_sizes_list[i]
                    scale_factors = scale_factors_list[i]
                    self.main_branch.add(ESPBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_sizes=kernel_sizes,
                        scale_factors=scale_factors,
                        use_residual=use_residual,
                        in_size=((in_size[0] // 2, in_size[1] // 2) if in_size else None),
                        bn_epsilon=bn_epsilon))
                    in_channels = out_channels

            self.preactiv = PreActivation(
                in_channels=(down_channels + in_channels),
                bn_epsilon=bn_epsilon)

    def hybrid_forward(self, F, x):
        x = self.down_conv(x)
        y = self.main_branch(x)
        x = F.concat(x, y, dim=1)
        x = self.preactiv(x)
        return x, y


class SBEncoderInitBlock(HybridBlock):
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
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bn_epsilon,
                 **kwargs):
        super(SBEncoderInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=2,
                bn_epsilon=bn_epsilon,
                activation=(lambda: PReLU2(mid_channels)))
            self.conv2 = dwsconv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=2,
                dw_use_bn=False,
                bn_epsilon=bn_epsilon,
                dw_activation=None,
                pw_activation=(lambda: PReLU2(out_channels)),
                se_reduction=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SBEncoder(HybridBlock):
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
    in_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_epsilon : float
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
                 in_size,
                 bn_epsilon,
                 **kwargs):
        super(SBEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.init_block = SBEncoderInitBlock(
                in_channels=in_channels,
                mid_channels=init_block_channels[0],
                out_channels=init_block_channels[1],
                bn_epsilon=bn_epsilon)

            in_channels = init_block_channels[1]
            self.stage1 = SBStage(
                in_channels=in_channels,
                down_channels=down_channels_list[0],
                channels_list=channels_list[0],
                kernel_sizes_list=kernel_sizes_list[0],
                scale_factors_list=scale_factors_list[0],
                use_residual_list=use_residual_list[0],
                se_reduction=1,
                in_size=((in_size[0] // 4, in_size[1] // 4) if in_size else None),
                bn_epsilon=bn_epsilon)

            in_channels = down_channels_list[0] + channels_list[0][-1]
            self.stage2 = SBStage(
                in_channels=in_channels,
                down_channels=down_channels_list[1],
                channels_list=channels_list[1],
                kernel_sizes_list=kernel_sizes_list[1],
                scale_factors_list=scale_factors_list[1],
                use_residual_list=use_residual_list[1],
                se_reduction=2,
                in_size=((in_size[0] // 8, in_size[1] // 8) if in_size else None),
                bn_epsilon=bn_epsilon)

            in_channels = down_channels_list[1] + channels_list[1][-1]
            self.output = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def hybrid_forward(self, F, x):
        y1 = self.init_block(x)
        x, y2 = self.stage1(y1)
        x, _ = self.stage2(x)
        x = self.output(x)
        return x, y2, y1


class SBDecodeBlock(HybridBlock):
    """
    SB decoder block for SINet.

    Parameters:
    ----------
    channels : int
        Number of output classes.
    out_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 out_size,
                 bn_epsilon,
                 **kwargs):
        super(SBDecodeBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.up = InterpolationBlock(
                scale_factor=2,
                out_size=out_size)
            self.bn = nn.BatchNorm(
                in_channels=channels,
                epsilon=bn_epsilon)

    def hybrid_forward(self, F, x, y):
        x = self.up(x)
        x = self.bn(x)
        w_conf = x.softmax()
        w_max = w_conf.max(axis=1).expand_dims(1).broadcast_like(x)
        x = y * (1 - w_max) + x
        return x


class SBDecoder(HybridBlock):
    """
    SB decoder for SINet.

    Parameters:
    ----------
    dim2 : int
        Size of dimension #2.
    classes : int
        Number of segmentation classes.
    out_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 dim2,
                 classes,
                 out_size,
                 bn_epsilon,
                 **kwargs):
        super(SBDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self.decode1 = SBDecodeBlock(
                channels=classes,
                out_size=((out_size[0] // 8, out_size[1] // 8) if out_size else None),
                bn_epsilon=bn_epsilon)
            self.decode2 = SBDecodeBlock(
                channels=classes,
                out_size=((out_size[0] // 4, out_size[1] // 4) if out_size else None),
                bn_epsilon=bn_epsilon)
            self.conv3c = conv1x1_block(
                in_channels=dim2,
                out_channels=classes,
                bn_epsilon=bn_epsilon,
                activation=(lambda: PReLU2(classes)))
            self.output = nn.Conv2DTranspose(
                channels=classes,
                kernel_size=2,
                strides=2,
                padding=0,
                output_padding=0,
                in_channels=classes,
                use_bias=False)
            self.up = InterpolationBlock(
                scale_factor=2,
                out_size=out_size)

    def hybrid_forward(self, F, y3, y2, y1):
        y2 = self.conv3c(y2)
        x = self.decode1(y3, y2)
        x = self.decode2(x, y1)
        x = self.output(x)
        x = self.up(x)
        return x


class SINet(HybridBlock):
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
    bn_epsilon : float
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 down_channels_list,
                 channels_list,
                 kernel_sizes_list,
                 scale_factors_list,
                 use_residual_list,
                 dim2,
                 bn_epsilon,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=21,
                 **kwargs):
        super(SINet, self).__init__(**kwargs)
        assert (fixed_size is not None)
        assert (in_channels > 0)
        assert ((in_size[0] % 64 == 0) and (in_size[1] % 64 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux

        with self.name_scope():
            init_block_channels = [16, classes]
            out_channels = classes
            self.encoder = SBEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                init_block_channels=init_block_channels,
                down_channels_list=down_channels_list,
                channels_list=channels_list,
                kernel_sizes_list=kernel_sizes_list,
                scale_factors_list=scale_factors_list,
                use_residual_list=use_residual_list,
                in_size=(in_size if fixed_size else None),
                bn_epsilon=bn_epsilon)

            self.decoder = SBDecoder(
                dim2=dim2,
                classes=classes,
                out_size=(in_size if fixed_size else None),
                bn_epsilon=bn_epsilon)

    def hybrid_forward(self, F, x):
        y3, y2, y1 = self.encoder(x)
        x = self.decoder(y3, y2, y1)
        if self.aux:
            return x, y3
        else:
            return x


def get_sinet(model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    """
    Create SINet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def sinet_cityscapes(classes=19, **kwargs):
    """
    SINet model for Cityscapes from 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze
    Modules and Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sinet(classes=classes, bn_epsilon=1e-3, model_name="sinet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (1024, 2048)
    aux = False
    fixed_size = True
    pretrained = False

    models = [
        sinet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux, fixed_size=fixed_size)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sinet_cityscapes or weight_count == 119418)

        batch = 14
        x = mx.nd.zeros((batch, 3, in_size[0], in_size[1]), ctx=ctx)
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
