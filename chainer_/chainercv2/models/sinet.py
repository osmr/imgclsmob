"""
    SINet for image segmentation, implemented in Chainer.
    Original paper: 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.
"""

__all__ = ['SINet', 'sinet_cityscapes']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, get_activation_layer, conv1x1_block, conv3x3_block, round_channels, dwconv_block,\
    Concurrent, ChannelShuffle, SimpleSequential


class InterpolationBlock(Chain):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    out_size : tuple of 2 int, default None
        Spatial size of the output tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 scale_factor,
                 out_size=None):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor
        self.out_size = out_size

    def __call__(self, x):
        out_size = self.out_size if (self.out_size is not None) else\
            (x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor)
        return F.resize_images(x, output_shape=out_size)


class SEBlock(Chain):
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
                 mid_activation=(lambda: F.relu),
                 out_activation=(lambda: F.sigmoid)):
        super(SEBlock, self).__init__()
        self.use_conv2 = (reduction > 1)
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        with self.init_scope():
            self.fc1 = L.Linear(
                in_size=channels,
                out_size=mid_channels)
            if self.use_conv2:
                self.activ = get_activation_layer(mid_activation)
                self.fc2 = L.Linear(
                    in_size=mid_channels,
                    out_size=channels)
            self.sigmoid = get_activation_layer(out_activation)

    def __call__(self, x):
        w = F.average_pooling_2d(x, ksize=x.shape[2:])
        w = self.fc1(w)
        if self.use_conv2:
            w = self.activ(w)
            w = self.fc2(w)
        w = self.sigmoid(w)
        w = F.broadcast_to(F.expand_dims(F.expand_dims(w, axis=2), axis=3), x.shape)
        x = x * w
        return x


class DwsConvBlock(Chain):
    """
    SINet version of depthwise separable convolution block with BatchNorms and activations at each convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    use_bias : bool, default False
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
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_eps=1e-5,
                 dw_activation=(lambda: F.relu),
                 pw_activation=(lambda: F.relu),
                 se_reduction=0):
        super(DwsConvBlock, self).__init__()
        self.use_se = (se_reduction > 0)

        with self.init_scope():
            self.dw_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                use_bias=use_bias,
                use_bn=dw_use_bn,
                bn_eps=bn_eps,
                activation=dw_activation)
            if self.use_se:
                self.se = SEBlock(
                    channels=in_channels,
                    reduction=se_reduction,
                    round_mid=False,
                    mid_activation=(lambda: L.PReLU(shape=(in_channels // se_reduction,))),
                    out_activation=(lambda: L.PReLU(shape=(in_channels,))))
            self.pw_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=pw_use_bn,
                bn_eps=bn_eps,
                activation=pw_activation)

    def __call__(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     stride=1,
                     pad=1,
                     dilate=1,
                     use_bias=False,
                     dw_use_bn=True,
                     pw_use_bn=True,
                     bn_eps=1e-5,
                     dw_activation=(lambda: F.relu),
                     pw_activation=(lambda: F.relu),
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
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    use_bias : bool, default False
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
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        dw_use_bn=dw_use_bn,
        pw_use_bn=pw_use_bn,
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        se_reduction=se_reduction)


def dwconv3x3_block(in_channels,
                    out_channels,
                    stride=1,
                    pad=1,
                    dilate=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation=(lambda: F.relu)):
    """
    3x3 depthwise version of the standard convolution block (SINet version).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation)


class FDWConvBlock(Chain):
    """
    Factorized depthwise separable convolution block with BatchNorms and activations at each convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    use_bias : bool, default False
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
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu)):
        super(FDWConvBlock, self).__init__()
        assert use_bn
        self.activate = (activation is not None)

        with self.init_scope():
            self.v_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=(ksize, 1),
                stride=stride,
                pad=(pad, 0),
                dilate=dilate,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_eps=bn_eps,
                activation=None)
            self.h_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=(1, ksize),
                stride=stride,
                pad=(0, pad),
                dilate=dilate,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_eps=bn_eps,
                activation=None)
            if self.activate:
                self.act = get_activation_layer(activation)

    def __call__(self, x):
        x = self.v_conv(x) + self.h_conv(x)
        if self.activate:
            x = self.act(x)
        return x


def fdwconv3x3_block(in_channels,
                     out_channels,
                     stride=1,
                     pad=1,
                     dilate=1,
                     use_bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation=(lambda: F.relu)):
    """
    3x3 factorized depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int, default 1
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    use_bias : bool, default False
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
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def fdwconv5x5_block(in_channels,
                     out_channels,
                     stride=1,
                     pad=2,
                     dilate=1,
                     use_bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation=(lambda: F.relu)):
    """
    5x5 factorized depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int, default 1
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
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
        ksize=5,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class SBBlock(Chain):
    """
    SB-block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int
        Convolution window size for a factorized depthwise separable convolution block.
    scale_factor : int
        Scale factor.
    size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 scale_factor,
                 size,
                 bn_eps):
        super(SBBlock, self).__init__()
        self.use_scale = (scale_factor > 1)

        with self.init_scope():
            if self.use_scale:
                self.down_scale = partial(
                    F.average_pooling_2d,
                    ksize=scale_factor,
                    stride=scale_factor)
                self.up_scale = InterpolationBlock(
                    scale_factor=scale_factor,
                    out_size=size)

            use_fdw = (scale_factor > 0)
            if use_fdw:
                fdwconv3x3_class = fdwconv3x3_block if ksize == 3 else fdwconv5x5_block
                self.conv1 = fdwconv3x3_class(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    bn_eps=bn_eps,
                    activation=(lambda: L.PReLU(shape=(in_channels,))))
            else:
                self.conv1 = dwconv3x3_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    bn_eps=bn_eps,
                    activation=(lambda: L.PReLU(shape=(in_channels,))))

            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=bn_eps)

    def __call__(self, x):
        if self.use_scale:
            x = self.down_scale(x)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_scale:
            x = self.up_scale(x)

        x = self.bn(x)
        return x


class PreActivation(Chain):
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
        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=bn_eps)
            self.activ = L.PReLU(shape=(in_channels,))

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ESPBlock(Chain):
    """
    ESP block, which is based on the following principle: Reduce ---> Split ---> Transform --> Merge.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksizes : list of int
        Convolution window size for branches.
    scale_factors : list of int
        Scale factor for branches.
    use_residual : bool
        Whether to use residual connection.
    in_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksizes,
                 scale_factors,
                 use_residual,
                 in_size,
                 bn_eps):
        super(ESPBlock, self).__init__()
        self.use_residual = use_residual
        groups = len(ksizes)

        mid_channels = int(out_channels / groups)
        res_channels = out_channels - groups * mid_channels

        with self.init_scope():
            self.conv = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=groups)

            self.c_shuffle = ChannelShuffle(
                channels=mid_channels,
                groups=groups)

            self.branches = Concurrent()
            with self.branches.init_scope():
                for i in range(groups):
                    out_channels_i = (mid_channels + res_channels) if i == 0 else mid_channels
                    setattr(self.branches, "branch{}".format(i + 1), SBBlock(
                        in_channels=mid_channels,
                        out_channels=out_channels_i,
                        ksize=ksizes[i],
                        scale_factor=scale_factors[i],
                        size=in_size,
                        bn_eps=bn_eps))

            self.preactiv = PreActivation(
                in_channels=out_channels,
                bn_eps=bn_eps)

    def __call__(self, x):
        if self.use_residual:
            identity = x

        x = self.conv(x)
        x = self.c_shuffle(x)
        x = self.branches(x)

        if self.use_residual:
            x = identity + x

        x = self.preactiv(x)
        return x


class SBStage(Chain):
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
    ksizes_list : list of int
        Convolution window size for branches.
    scale_factors_list : list of int
        Scale factor for branches.
    use_residual_list : list of int
        List of flags for using residual in each ESP-block.
    se_reduction : int
        Squeeze reduction value (0 means no-se).
    in_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 down_channels,
                 channels_list,
                 ksizes_list,
                 scale_factors_list,
                 use_residual_list,
                 se_reduction,
                 in_size,
                 bn_eps):
        super(SBStage, self).__init__()
        with self.init_scope():
            self.down_conv = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=down_channels,
                stride=2,
                dw_use_bn=False,
                bn_eps=bn_eps,
                dw_activation=None,
                pw_activation=(lambda: L.PReLU(shape=(down_channels,))),
                se_reduction=se_reduction)
            in_channels = down_channels

            self.main_branch = SimpleSequential()
            with self.main_branch.init_scope():
                for i, out_channels in enumerate(channels_list):
                    use_residual = (use_residual_list[i] == 1)
                    ksizes = ksizes_list[i]
                    scale_factors = scale_factors_list[i]
                    setattr(self.main_branch, "block{}".format(i + 1), ESPBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksizes=ksizes,
                        scale_factors=scale_factors,
                        use_residual=use_residual,
                        in_size=((in_size[0] // 2, in_size[1] // 2) if in_size else None),
                        bn_eps=bn_eps))
                    in_channels = out_channels

            self.preactiv = PreActivation(
                in_channels=(down_channels + in_channels),
                bn_eps=bn_eps)

    def __call__(self, x):
        x = self.down_conv(x)
        y = self.main_branch(x)
        x = F.concat((x, y), axis=1)
        x = self.preactiv(x)
        return x, y


class SBEncoderInitBlock(Chain):
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bn_eps):
        super(SBEncoderInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(shape=(mid_channels,))))
            self.conv2 = dwsconv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                stride=2,
                dw_use_bn=False,
                bn_eps=bn_eps,
                dw_activation=None,
                pw_activation=(lambda: L.PReLU(shape=(out_channels,))),
                se_reduction=1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SBEncoder(Chain):
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
    ksizes_list : list of list of int
        Convolution window size for each residual block.
    scale_factors_list : list of list of int
        Scale factor for each residual block.
    use_residual_list : list of list of int
        List of flags for using residual in each residual block.
    in_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 init_block_channels,
                 down_channels_list,
                 channels_list,
                 ksizes_list,
                 scale_factors_list,
                 use_residual_list,
                 in_size,
                 bn_eps):
        super(SBEncoder, self).__init__()
        with self.init_scope():
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
                ksizes_list=ksizes_list[0],
                scale_factors_list=scale_factors_list[0],
                use_residual_list=use_residual_list[0],
                se_reduction=1,
                in_size=((in_size[0] // 4, in_size[1] // 4) if in_size else None),
                bn_eps=bn_eps)

            in_channels = down_channels_list[0] + channels_list[0][-1]
            self.stage2 = SBStage(
                in_channels=in_channels,
                down_channels=down_channels_list[1],
                channels_list=channels_list[1],
                ksizes_list=ksizes_list[1],
                scale_factors_list=scale_factors_list[1],
                use_residual_list=use_residual_list[1],
                se_reduction=2,
                in_size=((in_size[0] // 8, in_size[1] // 8) if in_size else None),
                bn_eps=bn_eps)

            in_channels = down_channels_list[1] + channels_list[1][-1]
            self.output = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        y1 = self.init_block(x)
        x, y2 = self.stage1(y1)
        x, _ = self.stage2(x)
        x = self.output(x)
        return x, y2, y1


class SBDecodeBlock(Chain):
    """
    SB decoder block for SINet.

    Parameters:
    ----------
    channels : int
        Number of output classes.
    out_size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 bn_eps,
                 out_size,
                 channels):
        super(SBDecodeBlock, self).__init__()
        with self.init_scope():
            self.up = InterpolationBlock(
                scale_factor=2,
                out_size=out_size)
            self.bn = L.BatchNormalization(
                size=channels,
                eps=bn_eps)

    def __call__(self, x, y):
        x = self.up(x)
        x = self.bn(x)
        w_conf = F.softmax(x)
        w_max = F.broadcast_to(F.expand_dims(F.max(w_conf, axis=1), axis=1), x.shape)
        x = y * (1 - w_max) + x
        return x


class SBDecoder(Chain):
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
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 dim2,
                 classes,
                 out_size,
                 bn_eps):
        super(SBDecoder, self).__init__()
        with self.init_scope():
            self.decode1 = SBDecodeBlock(
                channels=classes,
                out_size=((out_size[0] // 8, out_size[1] // 8) if out_size else None),
                bn_eps=bn_eps)
            self.decode2 = SBDecodeBlock(
                channels=classes,
                out_size=((out_size[0] // 4, out_size[1] // 4) if out_size else None),
                bn_eps=bn_eps)
            self.conv3c = conv1x1_block(
                in_channels=dim2,
                out_channels=classes,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(shape=(classes,))))
            self.output = L.Deconvolution2D(
                in_channels=classes,
                out_channels=classes,
                ksize=2,
                stride=2,
                pad=0,
                # output_pad=0,
                nobias=True)
            self.up = InterpolationBlock(scale_factor=2)

    def __call__(self, y3, y2, y1):
        y2 = self.conv3c(y2)
        x = self.decode1(y3, y2)
        x = self.decode2(x, y1)
        x = self.output(x)
        x = self.up(x)
        return x


class SINet(Chain):
    """
    SINet model from 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.

    Parameters:
    ----------
    down_channels_list : list of int
        Number of downsample channels for each residual block.
    channels_list : list of list of int
        Number of output channels for all residual block.
    ksizes_list : list of list of int
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
    classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 down_channels_list,
                 channels_list,
                 ksizes_list,
                 scale_factors_list,
                 use_residual_list,
                 dim2,
                 bn_eps,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=21):
        super(SINet, self).__init__()
        assert (fixed_size is not None)
        assert (in_channels > 0)
        assert ((in_size[0] % 64 == 0) and (in_size[1] % 64 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux

        with self.init_scope():
            init_block_channels = [16, classes]
            out_channels = classes
            self.encoder = SBEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                init_block_channels=init_block_channels,
                down_channels_list=down_channels_list,
                channels_list=channels_list,
                ksizes_list=ksizes_list,
                scale_factors_list=scale_factors_list,
                use_residual_list=use_residual_list,
                in_size=(in_size if fixed_size else None),
                bn_eps=bn_eps)

            self.decoder = SBDecoder(
                dim2=dim2,
                classes=classes,
                out_size=(in_size if fixed_size else None),
                bn_eps=bn_eps)

    def __call__(self, x):
        y3, y2, y1 = self.encoder(x)
        x = self.decoder(y3, y2, y1)
        if self.aux:
            return x, y3
        else:
            return x


def get_sinet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".chainer", "models"),
              **kwargs):
    """
    Create SINet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    ksizes_list = [
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

    p = len(ksizes_list[0])
    q = len(ksizes_list[1])

    channels_list = [[dim2] * p, ([dim3] * (q // 2)) + ([dim4] * (q - q // 2))]
    use_residual_list = [[0] + ([1] * (p - 1)), [0] + ([1] * (q // 2 - 1)) + [0] + ([1] * (q - q // 2 - 1))]

    down_channels_list = [dim1, dim2]

    net = SINet(
        down_channels_list=down_channels_list,
        channels_list=channels_list,
        ksizes_list=ksizes_list,
        scale_factors_list=scale_factors_list,
        use_residual_list=use_residual_list,
        dim2=dims[1],
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sinet(classes=classes, bn_eps=1e-3, model_name="sinet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (1024, 2048)
    aux = False
    fixed_size = False
    pretrained = False

    models = [
        sinet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, aux=aux, fixed_size=fixed_size)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sinet_cityscapes or weight_count == 119418)

        batch = 14
        x = np.zeros((batch, 3, in_size[0], in_size[1]), np.float32)
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
