"""
    SINet for image segmentation, implemented in TensorFlow.
    Original paper: 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.
"""

__all__ = ['SINet', 'sinet_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import PReLU2, BatchNorm, AvgPool2d, conv1x1, get_activation_layer, conv1x1_block, conv3x3_block,\
    round_channels, dwconv_block, InterpolationBlock, ChannelShuffle, SimpleSequential, Concurrent, get_channel_axis,\
    is_channels_first


class SEBlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 round_mid=False,
                 mid_activation="relu",
                 out_activation="sigmoid",
                 data_format="channels_last",
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.data_format = data_format
        self.use_conv2 = (reduction > 1)
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.fc1 = nn.Dense(
            units=mid_channels,
            input_dim=channels,
            name="fc1")
        if self.use_conv2:
            self.activ = get_activation_layer(mid_activation, name="activ")
            self.fc2 = nn.Dense(
                units=channels,
                input_dim=mid_channels,
                name="fc2")
        self.sigmoid = get_activation_layer(out_activation, name="sigmoid")

    def call(self, x, training=None):
        w = self.pool(x)
        w = self.fc1(w)
        if self.use_conv2:
            w = self.activ(w)
            w = self.fc2(w)
        w = self.sigmoid(w)
        axis = -1 if is_channels_first(self.data_format) else 1
        w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        x = x * w
        return x


class DwsConvBlock(nn.Layer):
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default 'relu'
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default 'relu'
        Activation function after the pointwise convolution block.
    se_reduction : int, default 0
        Squeeze reduction value (0 means no-se).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 bn_eps=1e-5,
                 dw_activation="relu",
                 pw_activation="relu",
                 se_reduction=0,
                 data_format="channels_last",
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        self.use_se = (se_reduction > 0)

        self.dw_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            use_bias=use_bias,
            use_bn=dw_use_bn,
            bn_eps=bn_eps,
            activation=dw_activation,
            data_format=data_format,
            name="dw_conv")
        if self.use_se:
            self.se = SEBlock(
                channels=in_channels,
                reduction=se_reduction,
                round_mid=False,
                mid_activation=(lambda: PReLU2(in_channels // se_reduction, data_format=data_format, name="activ")),
                out_activation=(lambda: PReLU2(in_channels, data_format=data_format, name="sigmoid")),
                data_format=data_format,
                name="se")
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            use_bn=pw_use_bn,
            bn_eps=bn_eps,
            activation=pw_activation,
            data_format=data_format,
            name="pw_conv")

    def call(self, x, training=None):
        x = self.dw_conv(x, training=None)
        if self.use_se:
            x = self.se(x, training=None)
        x = self.pw_conv(x, training=None)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=1,
                     dilation=1,
                     use_bias=False,
                     dw_use_bn=True,
                     pw_use_bn=True,
                     bn_eps=1e-5,
                     dw_activation="relu",
                     pw_activation="relu",
                     se_reduction=0,
                     data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default 'relu'
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default 'relu'
        Activation function after the pointwise convolution block.
    se_reduction : int, default 0
        Squeeze reduction value (0 means no-se).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        se_reduction=se_reduction,
        data_format=data_format,
        **kwargs)


def dwconv3x3_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    bn_eps=1e-5,
                    activation="relu",
                    data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


class FDWConvBlock(nn.Layer):
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function after the each convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(FDWConvBlock, self).__init__(**kwargs)
        assert use_bn
        self.activate = (activation is not None)

        self.v_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            strides=strides,
            padding=(padding, 0),
            dilation=dilation,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=None,
            data_format=data_format,
            name="v_conv")
        self.h_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            strides=strides,
            padding=(0, padding),
            dilation=dilation,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=None,
            data_format=data_format,
            name="h_conv")
        if self.activate:
            self.act = get_activation_layer(activation, name="act")

    def call(self, x, training=None):
        x = self.v_conv(x, training=None) + self.h_conv(x, training=None)
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
                     bn_eps=1e-5,
                     activation="relu",
                     data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


def fdwconv5x5_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=2,
                     dilation=1,
                     use_bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation="relu",
                     data_format="channels_last",
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        **kwargs)


class SBBlock(nn.Layer):
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
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor,
                 size,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(SBBlock, self).__init__(**kwargs)
        self.use_scale = (scale_factor > 1)

        if self.use_scale:
            self.down_scale = AvgPool2d(
                pool_size=scale_factor,
                strides=scale_factor,
                data_format=data_format,
                name="down_scale")
            self.up_scale = InterpolationBlock(
                scale_factor=scale_factor,
                out_size=size,
                data_format=data_format,
                name="up_scale")

        use_fdw = (scale_factor > 0)
        if use_fdw:
            fdwconv3x3_class = fdwconv3x3_block if kernel_size == 3 else fdwconv5x5_block
            self.conv1 = fdwconv3x3_class(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_eps=bn_eps,
                activation=(lambda: PReLU2(in_channels, data_format=data_format, name="activ")),
                data_format=data_format,
                name="conv1")
        else:
            self.conv1 = dwconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_eps=bn_eps,
                activation=(lambda: PReLU2(in_channels, data_format=data_format, name="activ")),
                data_format=data_format,
                name="conv1")

        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv2")

        self.bn = BatchNorm(
            epsilon=bn_eps,
            data_format=data_format,
            name="bn")

    def call(self, x, training=None):
        if self.use_scale:
            x = self.down_scale(x)

        x = self.conv1(x, training=None)
        x = self.conv2(x, training=None)

        if self.use_scale:
            x = self.up_scale(x)

        x = self.bn(x, training=None)
        return x


class PreActivation(nn.Layer):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 bn_eps=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(PreActivation, self).__init__(**kwargs)
        assert (in_channels is not None)

        self.bn = BatchNorm(
            epsilon=bn_eps,
            data_format=data_format,
            name="bn")
        self.activ = PReLU2(in_channels, data_format=data_format, name="activ")

    def call(self, x, training=None):
        x = self.bn(x, training=None)
        x = self.activ(x)
        return x


class ESPBlock(nn.Layer):
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
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 scale_factors,
                 use_residual,
                 in_size,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(ESPBlock, self).__init__(**kwargs)
        self.use_residual = use_residual
        groups = len(kernel_sizes)

        mid_channels = int(out_channels / groups)
        res_channels = out_channels - groups * mid_channels

        self.conv = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=groups,
            data_format=data_format,
            name="conv")

        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=groups,
            data_format=data_format,
            name="c_shuffle")

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        for i in range(groups):
            out_channels_i = (mid_channels + res_channels) if i == 0 else mid_channels
            self.branches.add(SBBlock(
                in_channels=mid_channels,
                out_channels=out_channels_i,
                kernel_size=kernel_sizes[i],
                scale_factor=scale_factors[i],
                size=in_size,
                bn_eps=bn_eps,
                data_format=data_format,
                name="branch{}".format(i + 1)))

        self.preactiv = PreActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            data_format=data_format,
            name="preactiv")

    def call(self, x, training=None):
        if self.use_residual:
            identity = x

        x = self.conv(x)
        x = self.c_shuffle(x)
        x = self.branches(x, training=None)

        if self.use_residual:
            x = identity + x

        x = self.preactiv(x, training=None)
        return x


class SBStage(nn.Layer):
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
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(SBStage, self).__init__(**kwargs)
        self.data_format = data_format

        self.down_conv = dwsconv3x3_block(
            in_channels=in_channels,
            out_channels=down_channels,
            strides=2,
            dw_use_bn=False,
            bn_eps=bn_eps,
            dw_activation=None,
            pw_activation=(lambda: PReLU2(down_channels, data_format=data_format, name="activ")),
            se_reduction=se_reduction,
            data_format=data_format,
            name="down_conv")
        in_channels = down_channels

        self.main_branch = SimpleSequential(name="main_branch")
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
                bn_eps=bn_eps,
                data_format=data_format,
                name="block{}".format(i + 1)))
            in_channels = out_channels

        self.preactiv = PreActivation(
            in_channels=(down_channels + in_channels),
            bn_eps=bn_eps,
            data_format=data_format,
            name="preactiv")

    def call(self, x, training=None):
        x = self.down_conv(x, training=None)
        y = self.main_branch(x, training=None)
        x = tf.concat([x, y], axis=get_channel_axis(self.data_format))
        x = self.preactiv(x, training=None)
        return x, y


class SBEncoderInitBlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(SBEncoderInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(mid_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv1")
        self.conv2 = dwsconv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=2,
            dw_use_bn=False,
            bn_eps=bn_eps,
            dw_activation=None,
            pw_activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            se_reduction=1,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=None)
        x = self.conv2(x, training=None)
        return x


class SBEncoder(nn.Layer):
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
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(SBEncoder, self).__init__(**kwargs)
        self.init_block = SBEncoderInitBlock(
            in_channels=in_channels,
            mid_channels=init_block_channels[0],
            out_channels=init_block_channels[1],
            bn_eps=bn_eps,
            data_format=data_format,
            name="init_block")

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
            bn_eps=bn_eps,
            data_format=data_format,
            name="stage1")

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
            bn_eps=bn_eps,
            data_format=data_format,
            name="stage2")

        in_channels = down_channels_list[1] + channels_list[1][-1]
        self.output_conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="output")

    def call(self, x, training=None):
        y1 = self.init_block(x, training=None)
        x, y2 = self.stage1(y1, training=None)
        x, _ = self.stage2(x, training=None)
        x = self.output_conv(x)
        return x, y2, y1


class SBDecodeBlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 out_size,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(SBDecodeBlock, self).__init__(**kwargs)
        assert (channels is not None)
        self.data_format = data_format

        self.up = InterpolationBlock(
            scale_factor=2,
            out_size=out_size,
            data_format=data_format,
            name="up")
        self.bn = BatchNorm(
            epsilon=bn_eps,
            data_format=data_format,
            name="bn")

    def call(self, x, y, training=None):
        x = self.up(x)
        x = self.bn(x, training=None)
        w_conf = tf.nn.softmax(x)
        axis = get_channel_axis(self.data_format)
        w_max = tf.broadcast_to(tf.expand_dims(tf.reduce_max(w_conf, axis=axis), axis=axis), shape=x.shape)
        x = y * (1 - w_max) + x
        return x


class SBDecoder(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 dim2,
                 classes,
                 out_size,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(SBDecoder, self).__init__(**kwargs)
        self.decode1 = SBDecodeBlock(
            channels=classes,
            out_size=((out_size[0] // 8, out_size[1] // 8) if out_size else None),
            bn_eps=bn_eps,
            data_format=data_format,
            name="decode1")
        self.decode2 = SBDecodeBlock(
            channels=classes,
            out_size=((out_size[0] // 4, out_size[1] // 4) if out_size else None),
            bn_eps=bn_eps,
            data_format=data_format,
            name="decode2")
        self.conv3c = conv1x1_block(
            in_channels=dim2,
            out_channels=classes,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(classes, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv3c")
        self.output_conv = nn.Conv2DTranspose(
            filters=classes,
            kernel_size=2,
            strides=2,
            padding="valid",
            output_padding=0,
            use_bias=False,
            data_format=data_format,
            name="output_conv")
        self.up = InterpolationBlock(
            scale_factor=2,
            out_size=out_size,
            data_format=data_format,
            name="up")

    def call(self, y3, y2, y1, training=None):
        y2 = self.conv3c(y2, training=None)
        x = self.decode1(y3, y2, training=None)
        x = self.decode2(x, y1, training=None)
        x = self.output_conv(x, training=None)
        x = self.up(x)
        return x


class SINet(tf.keras.Model):
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
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    classes : int, default 21
        Number of segmentation classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 fixed_size=True,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=21,
                 data_format="channels_last",
                 **kwargs):
        super(SINet, self).__init__(**kwargs)
        assert (fixed_size is not None)
        assert (in_channels > 0)
        assert ((in_size[0] % 64 == 0) and (in_size[1] % 64 == 0))
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        self.aux = aux

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
            bn_eps=bn_eps,
            data_format=data_format,
            name="encoder")

        self.decoder = SBDecoder(
            dim2=dim2,
            classes=classes,
            out_size=(in_size if fixed_size else None),
            bn_eps=bn_eps,
            data_format=data_format,
            name="decoder")

    def call(self, x, training=None):
        y3, y2, y1 = self.encoder(x, training=None)
        x = self.decoder(y3, y2, y1, training=None)
        if self.aux:
            return x, y3
        else:
            return x


def get_sinet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create SINet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

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
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_sinet(classes=classes, bn_eps=1e-3, model_name="sinet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (1024, 2048)
    aux = False
    fixed_size = False
    pretrained = False

    models = [
        sinet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux, fixed_size=fixed_size)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape[0] == x.shape[0])
        if is_channels_first(data_format):
            assert ((y.shape[1] == 19) and (y.shape[2] == x.shape[2]) and (y.shape[3] == x.shape[3]))
        else:
            assert ((y.shape[3] == 19) and (y.shape[1] == x.shape[1]) and (y.shape[2] == x.shape[2]))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sinet_cityscapes or weight_count == 119418)


if __name__ == "__main__":
    _test()
