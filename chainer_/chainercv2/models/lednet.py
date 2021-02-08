"""
    LEDNet for image segmentation, implemented in Chainer.
    Original paper: 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.
"""

__all__ = ['LEDNet', 'lednet_cityscapes']

import os
import chainer.functions as F
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, conv1x1_block, conv3x3_block, conv5x5_block, conv7x7_block, ConvBlock, NormActivation,\
    ChannelShuffle, InterpolationBlock, Hourglass, BreakBlock, SimpleSequential


class AsymConvBlock(Chain):
    """
    Asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    ksize : int
        Convolution window size.
    pad : int
        Padding value for convolution layer.
    dilate : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    lw_use_bn : bool, default True
        Whether to use BatchNorm layer (leftwise convolution block).
    rw_use_bn : bool, default True
        Whether to use BatchNorm layer (rightwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    lw_activation : function or str or None, default F.relu
        Activation function after the leftwise convolution block.
    rw_activation : function or str or None, default F.relu
        Activation function after the rightwise convolution block.
    """
    def __init__(self,
                 channels,
                 ksize,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 lw_use_bn=True,
                 rw_use_bn=True,
                 bn_eps=1e-5,
                 lw_activation=(lambda: F.relu),
                 rw_activation=(lambda: F.relu)):
        super(AsymConvBlock, self).__init__()
        with self.init_scope():
            self.lw_conv = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                ksize=(ksize, 1),
                stride=1,
                pad=(pad, 0),
                dilate=(dilate, 1),
                groups=groups,
                use_bias=use_bias,
                use_bn=lw_use_bn,
                bn_eps=bn_eps,
                activation=lw_activation)
            self.rw_conv = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                ksize=(1, ksize),
                stride=1,
                pad=(0, pad),
                dilate=(1, dilate),
                groups=groups,
                use_bias=use_bias,
                use_bn=rw_use_bn,
                bn_eps=bn_eps,
                activation=rw_activation)

    def __call__(self, x):
        x = self.lw_conv(x)
        x = self.rw_conv(x)
        return x


def asym_conv3x3_block(pad=1,
                       **kwargs):
    """
    3x3 asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    pad : int
        Padding value for convolution layer.
    dilate : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    lw_use_bn : bool, default True
        Whether to use BatchNorm layer (leftwise convolution block).
    rw_use_bn : bool, default True
        Whether to use BatchNorm layer (rightwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    lw_activation : function or str or None, default F.relu
        Activation function after the leftwise convolution block.
    rw_activation : function or str or None, default F.relu
        Activation function after the rightwise convolution block.
    """
    return AsymConvBlock(
        ksize=3,
        pad=pad,
        **kwargs)


class LEDDownBlock(Chain):
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

        with self.init_scope():
            self.pool = partial(
                F.max_pooling_2d,
                ksize=2,
                stride=2,
                cover_all=False)
            self.conv = conv3x3(
                in_channels=in_channels,
                out_channels=(out_channels - in_channels),
                stride=2,
                use_bias=True)
            self.norm_activ = NormActivation(
                in_channels=out_channels,
                bn_eps=bn_eps)

    def __call__(self, x):
        y1 = self.pool(x)
        y2 = self.conv(x)

        if self.correct_size_mismatch:
            diff_h = y2.size()[2] - y1.size()[2]
            diff_w = y2.size()[3] - y1.size()[3]
            y1 = F.pad(
                y1,
                pad_width=((0, 0), (0, 0), (diff_w // 2, diff_w - diff_w // 2), (diff_h // 2, diff_h - diff_h // 2)),
                mode="constant",
                constant_values=0)

        x = F.concat((y2, y1), axis=1)
        x = self.norm_activ(x)
        return x


class LEDBranch(Chain):
    """
    LEDNet encoder branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilate : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 dilate,
                 dropout_rate,
                 bn_eps):
        super(LEDBranch, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        with self.init_scope():
            self.conv1 = asym_conv3x3_block(
                channels=channels,
                use_bias=True,
                lw_use_bn=False,
                bn_eps=bn_eps)
            self.conv2 = asym_conv3x3_block(
                channels=channels,
                pad=dilate,
                dilate=dilate,
                use_bias=True,
                lw_use_bn=False,
                bn_eps=bn_eps,
                rw_activation=None)
            if self.use_dropout:
                self.dropout = partial(
                    F.dropout,
                    ratio=dropout_rate)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class LEDUnit(Chain):
    """
    LEDNet encoder unit (Split-Shuffle-non-bottleneck).

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilate : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 dilate,
                 dropout_rate,
                 bn_eps):
        super(LEDUnit, self).__init__()
        mid_channels = channels // 2

        with self.init_scope():
            self.left_branch = LEDBranch(
                channels=mid_channels,
                dilate=dilate,
                dropout_rate=dropout_rate,
                bn_eps=bn_eps)
            self.right_branch = LEDBranch(
                channels=mid_channels,
                dilate=dilate,
                dropout_rate=dropout_rate,
                bn_eps=bn_eps)
            self.activ = F.relu
            self.shuffle = ChannelShuffle(
                channels=channels,
                groups=2)

    def __call__(self, x):
        identity = x

        x1, x2 = F.split_axis(x, indices_or_sections=2, axis=1)
        x1 = self.left_branch(x1)
        x2 = self.right_branch(x2)
        x = F.concat((x1, x2), axis=1)

        x = x + identity
        x = self.activ(x)
        x = self.shuffle(x)
        return x


class PoolingBranch(Chain):
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
    bn_eps : float
        Small float added to variance in Batch norm.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    down_size : int
        Spatial size of downscaled image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias,
                 bn_eps,
                 in_size,
                 down_size):
        super(PoolingBranch, self).__init__()
        self.in_size = in_size
        self.down_size = down_size

        with self.init_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                bn_eps=bn_eps)
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=in_size)

    def __call__(self, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = F.average_pooling_2d(x, ksize=(in_size[0] // self.down_size, in_size[1] // self.down_size))
        x = self.conv(x)
        x = self.up(x, in_size)
        return x


class APN(Chain):
    """
    Attention pyramid network block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 in_size):
        super(APN, self).__init__()
        self.in_size = in_size
        att_out_channels = 1

        with self.init_scope():
            self.pool_branch = PoolingBranch(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_eps=bn_eps,
                in_size=in_size,
                down_size=1)

            self.body = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_eps=bn_eps)

            down_seq = SimpleSequential()
            with down_seq.init_scope():
                setattr(down_seq, "down1", conv7x7_block(
                    in_channels=in_channels,
                    out_channels=att_out_channels,
                    stride=2,
                    use_bias=True,
                    bn_eps=bn_eps))
                setattr(down_seq, "down2", conv5x5_block(
                    in_channels=att_out_channels,
                    out_channels=att_out_channels,
                    stride=2,
                    use_bias=True,
                    bn_eps=bn_eps))
                down3_subseq = SimpleSequential()
                with down3_subseq.init_scope():
                    setattr(down3_subseq, "conv1", conv3x3_block(
                        in_channels=att_out_channels,
                        out_channels=att_out_channels,
                        stride=2,
                        use_bias=True,
                        bn_eps=bn_eps))
                    setattr(down3_subseq, "conv2", conv3x3_block(
                        in_channels=att_out_channels,
                        out_channels=att_out_channels,
                        use_bias=True,
                        bn_eps=bn_eps))
                setattr(down_seq, "down3", down3_subseq)

            up_seq = SimpleSequential()
            with up_seq.init_scope():
                up = InterpolationBlock(scale_factor=2)
                setattr(up_seq, "up1", up)
                setattr(up_seq, "up2", up)
                setattr(up_seq, "up3", up)

            skip_seq = SimpleSequential()
            with skip_seq.init_scope():
                setattr(skip_seq, "skip1", BreakBlock())
                setattr(skip_seq, "skip2", conv7x7_block(
                    in_channels=att_out_channels,
                    out_channels=att_out_channels,
                    use_bias=True,
                    bn_eps=bn_eps))
                setattr(skip_seq, "skip3", conv5x5_block(
                    in_channels=att_out_channels,
                    out_channels=att_out_channels,
                    use_bias=True,
                    bn_eps=bn_eps))

            self.hg = Hourglass(
                down_seq=down_seq,
                up_seq=up_seq,
                skip_seq=skip_seq)

    def __call__(self, x):
        y = self.pool_branch(x)
        w = self.hg(x)
        x = self.body(x)
        x = x * w
        x = x + y
        return x


class LEDNet(Chain):
    """
    LEDNet model from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit.
    dilates : list of int
        Dilations for units.
    dropout_rates : list of list of int
        Dropout rates for each unit in encoder.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images in encoder.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 dilates,
                 dropout_rates,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19):
        super(LEDNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size

        with self.init_scope():
            self.encoder = SimpleSequential()
            with self.encoder.init_scope():
                for i, dilates_per_stage in enumerate(dilates):
                    out_channels = channels[i]
                    dropout_rate = dropout_rates[i]
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, dilate in enumerate(dilates_per_stage):
                            if j == 0:
                                setattr(stage, "unit{}".format(j + 1), LEDDownBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    correct_size_mismatch=correct_size_mismatch,
                                    bn_eps=bn_eps))
                                in_channels = out_channels
                            else:
                                setattr(stage, "unit{}".format(j + 1), LEDUnit(
                                    channels=in_channels,
                                    dilate=dilate,
                                    dropout_rate=dropout_rate,
                                    bn_eps=bn_eps))
                    setattr(self.encoder, "stage{}".format(i + 1), stage)
            self.apn = APN(
                in_channels=in_channels,
                out_channels=classes,
                bn_eps=bn_eps,
                in_size=(in_size[0] // 8, in_size[1] // 8) if fixed_size else None)
            self.up = InterpolationBlock(
                scale_factor=8,
                align_corners=True)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.apn(x)
        x = self.up(x)
        return x


def get_lednet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create LEDNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    channels = [32, 64, 128]
    dilates = [[0, 1, 1, 1], [0, 1, 1], [0, 1, 2, 5, 9, 2, 5, 9, 17]]
    dropout_rates = [0.03, 0.03, 0.3]
    bn_eps = 1e-3

    net = LEDNet(
        channels=channels,
        dilates=dilates,
        dropout_rates=dropout_rates,
        bn_eps=bn_eps,
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_lednet(classes=classes, model_name="lednet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

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
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lednet_cityscapes or weight_count == 922821)

        batch = 4
        x = np.random.rand(batch, 3, in_size[0], in_size[1]).astype(np.float32)
        y = net(x)
        assert (y.shape == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
