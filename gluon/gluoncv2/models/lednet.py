"""
    LEDNet for image segmentation, implemented in Gluon.
    Original paper: 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.
"""

__all__ = ['LEDNet', 'lednet_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3, conv1x1_block, conv3x3_block, conv5x5_block, conv7x7_block, ConvBlock, NormActivation,\
    ChannelShuffle, InterpolationBlock, Hourglass, BreakBlock


class AsymConvBlock(HybridBlock):
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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    lw_use_bn : bool, default True
        Whether to use BatchNorm layer (leftwise convolution block).
    rw_use_bn : bool, default True
        Whether to use BatchNorm layer (rightwise convolution block).
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    lw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the leftwise convolution block.
    rw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the rightwise convolution block.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 lw_use_bn=True,
                 rw_use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 lw_activation=(lambda: nn.Activation("relu")),
                 rw_activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(AsymConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.lw_conv = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(kernel_size, 1),
                strides=1,
                padding=(padding, 0),
                dilation=(dilation, 1),
                groups=groups,
                use_bias=use_bias,
                use_bn=lw_use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=lw_activation)
            self.rw_conv = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, kernel_size),
                strides=1,
                padding=(0, padding),
                dilation=(1, dilation),
                groups=groups,
                use_bias=use_bias,
                use_bn=rw_use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=rw_activation)

    def hybrid_forward(self, F, x):
        x = self.lw_conv(x)
        x = self.rw_conv(x)
        return x


def asym_conv3x3_block(padding=1,
                       **kwargs):
    """
    3x3 asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    lw_use_bn : bool, default True
        Whether to use BatchNorm layer (leftwise convolution block).
    rw_use_bn : bool, default True
        Whether to use BatchNorm layer (rightwise convolution block).
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    lw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the leftwise convolution block.
    rw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the rightwise convolution block.
    """
    return AsymConvBlock(
        kernel_size=3,
        padding=padding,
        **kwargs)


class LEDDownBlock(HybridBlock):
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
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 correct_size_mismatch,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(LEDDownBlock, self).__init__(**kwargs)
        self.correct_size_mismatch = correct_size_mismatch

        with self.name_scope():
            self.pool = nn.MaxPool2D(
                pool_size=2,
                strides=2)
            self.conv = conv3x3(
                in_channels=in_channels,
                out_channels=(out_channels - in_channels),
                strides=2,
                use_bias=True)
            self.norm_activ = NormActivation(
                in_channels=out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        y1 = self.pool(x)
        y2 = self.conv(x)

        if self.correct_size_mismatch:
            diff_h = y2.size()[2] - y1.size()[2]
            diff_w = y2.size()[3] - y1.size()[3]
            y1 = F.pad(
                y1,
                mode="constant",
                pad_width=(0, 0, 0, 0, diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2),
                constant_value=0)

        x = F.concat(y2, y1, dim=1)
        x = self.norm_activ(x)
        return x


class LEDBranch(HybridBlock):
    """
    LEDNet encoder branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(LEDBranch, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)

        with self.name_scope():
            self.conv1 = asym_conv3x3_block(
                channels=channels,
                use_bias=True,
                lw_use_bn=False,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2 = asym_conv3x3_block(
                channels=channels,
                padding=dilation,
                dilation=dilation,
                use_bias=True,
                lw_use_bn=False,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                rw_activation=None)
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class LEDUnit(HybridBlock):
    """
    LEDNet encoder unit (Split-Shuffle-non-bottleneck).

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(LEDUnit, self).__init__(**kwargs)
        mid_channels = channels // 2

        with self.name_scope():
            self.left_branch = LEDBranch(
                channels=mid_channels,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.right_branch = LEDBranch(
                channels=mid_channels,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.activ = nn.Activation("relu")
            self.shuffle = ChannelShuffle(
                channels=channels,
                groups=2)

    def hybrid_forward(self, F, x):
        identity = x

        x1, x2 = F.split(x, axis=1, num_outputs=2)
        x1 = self.left_branch(x1)
        x2 = self.right_branch(x2)
        x = F.concat(x1, x2, dim=1)

        x = x + identity
        x = self.activ(x)
        x = self.shuffle(x)
        return x


class PoolingBranch(HybridBlock):
    """
    Pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bias : bool
        Whether the layer uses a bias vector.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool
        Whether to disable CUDNN batch normalization operator.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    down_size : int
        Spatial size of downscaled image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias,
                 bn_epsilon,
                 bn_use_global_stats,
                 bn_cudnn_off,
                 in_size,
                 down_size,
                 **kwargs):
        super(PoolingBranch, self).__init__(**kwargs)
        self.in_size = in_size
        self.down_size = down_size

        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=in_size)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = F.contrib.AdaptiveAvgPooling2D(x, output_size=self.down_size)
        x = self.conv(x)
        x = self.up(x, in_size)
        return x


class APN(HybridBlock):
    """
    Attention pyramid network block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool
        Whether to disable CUDNN batch normalization operator.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_epsilon,
                 bn_use_global_stats,
                 bn_cudnn_off,
                 in_size,
                 **kwargs):
        super(APN, self).__init__(**kwargs)
        self.in_size = in_size
        att_out_channels = 1

        with self.name_scope():
            self.pool_branch = PoolingBranch(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                in_size=in_size,
                down_size=1)

            self.body = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

            down_seq = nn.HybridSequential(prefix="")
            down_seq.add(conv7x7_block(
                in_channels=in_channels,
                out_channels=att_out_channels,
                strides=2,
                use_bias=True,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            down_seq.add(conv5x5_block(
                in_channels=att_out_channels,
                out_channels=att_out_channels,
                strides=2,
                use_bias=True,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            down3_subseq = nn.HybridSequential(prefix="")
            down3_subseq.add(conv3x3_block(
                in_channels=att_out_channels,
                out_channels=att_out_channels,
                strides=2,
                use_bias=True,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            down3_subseq.add(conv3x3_block(
                in_channels=att_out_channels,
                out_channels=att_out_channels,
                use_bias=True,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            down_seq.add(down3_subseq)

        up_seq = nn.HybridSequential(prefix="")
        up = InterpolationBlock(scale_factor=2)
        up_seq.add(up)
        up_seq.add(up)
        up_seq.add(up)

        skip_seq = nn.HybridSequential(prefix="")
        skip_seq.add(BreakBlock())
        skip_seq.add(conv7x7_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            use_bias=True,
            bn_epsilon=bn_epsilon,
            bn_use_global_stats=bn_use_global_stats,
            bn_cudnn_off=bn_cudnn_off))
        skip_seq.add(conv5x5_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            use_bias=True,
            bn_epsilon=bn_epsilon,
            bn_use_global_stats=bn_use_global_stats,
            bn_cudnn_off=bn_cudnn_off))

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq)

    def hybrid_forward(self, F, x):
        y = self.pool_branch(x)
        w = self.hg(x)
        x = self.body(x)
        x = x * w
        x = x + y
        return x


class LEDNet(HybridBlock):
    """
    LEDNet model from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit.
    dilations : list of int
        Dilations for units.
    dropout_rates : list of list of int
        Dropout rates for each unit in encoder.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images in encoder.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 channels,
                 dilations,
                 dropout_rates,
                 correct_size_mismatch=False,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 **kwargs):
        super(LEDNet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size

        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix="")
            for i, dilations_per_stage in enumerate(dilations):
                out_channels = channels[i]
                dropout_rate = dropout_rates[i]
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                for j, dilation in enumerate(dilations_per_stage):
                    if j == 0:
                        stage.add(LEDDownBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            correct_size_mismatch=correct_size_mismatch,
                            bn_epsilon=bn_epsilon,
                            bn_use_global_stats=bn_use_global_stats,
                            bn_cudnn_off=bn_cudnn_off))
                        in_channels = out_channels
                    else:
                        stage.add(LEDUnit(
                            channels=in_channels,
                            dilation=dilation,
                            dropout_rate=dropout_rate,
                            bn_epsilon=bn_epsilon,
                            bn_use_global_stats=bn_use_global_stats,
                            bn_cudnn_off=bn_cudnn_off))
                self.encoder.add(stage)
            self.apn = APN(
                in_channels=in_channels,
                out_channels=classes,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                in_size=(in_size[0] // 8, in_size[1] // 8) if fixed_size else None)
            self.up = InterpolationBlock(scale_factor=8)

    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.apn(x)
        x = self.up(x)
        return x


def get_lednet(model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create LEDNet model with specific parameters.

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
    channels = [32, 64, 128]
    dilations = [[0, 1, 1, 1], [0, 1, 1], [0, 1, 2, 5, 9, 2, 5, 9, 17]]
    dropout_rates = [0.03, 0.03, 0.3]
    bn_epsilon = 1e-3

    net = LEDNet(
        channels=channels,
        dilations=dilations,
        dropout_rates=dropout_rates,
        bn_epsilon=bn_epsilon,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx,
            ignore_extra=True)

    return net


def lednet_cityscapes(classes=19, **kwargs):
    """
    LEDNet model for Cityscapes from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic
    Segmentation,' https://arxiv.org/abs/1905.02423.

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
    return get_lednet(classes=classes, model_name="lednet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def _test():
    import mxnet as mx

    pretrained = False
    fixed_size = True
    correct_size_mismatch = False
    in_size = (1024, 2048)
    classes = 19

    models = [
        lednet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size,
                    correct_size_mismatch=correct_size_mismatch)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lednet_cityscapes or weight_count == 922821)

        batch = 4
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (y.shape == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
