"""
    QuartzNet for ASR, implemented in Chainer.
    Original paper: 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions,'
    https://arxiv.org/abs/1910.10261.
"""

__all__ = ['QuartzNet', 'quartznet5x5', 'quartznet10x5', 'quartznet15x5']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential
from .jasper import conv1d1, conv1d1_block, ConvBlock1d


class ChannelShuffle1d(Chain):
    """
    1D version of the channel shuffle layer.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups,
                 **kwargs):
        super(ChannelShuffle1d, self).__init__(**kwargs)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def __call__(self, x):
        batch, channels, seq_len = x.shape
        channels_per_group = channels // self.groups
        x = F.reshape(x, shape=(batch, self.groups, channels_per_group, seq_len))
        x = F.swapaxes(x, axis1=1, axis2=2)
        x = F.reshape(x, shape=(batch, channels, seq_len))
        return x

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(
            name=self.__class__.__name__,
            groups=self.groups)


class DwsConvBlock1d(Chain):
    """
    Depthwise version of the 1D standard convolution block with batch normalization, activation, dropout, and channel
    shuffle.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int
        Convolution window size.
    stride : int
        Stride of the convolution.
    pad : int
        Padding value for convolution layer.
    dilate : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu),
                 dropout_rate=0.0,
                 **kwargs):
        super(DwsConvBlock1d, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_dropout = (dropout_rate != 0.0)
        self.use_channel_shuffle = (groups > 1)

        with self.init_scope():
            self.dw_conv = L.Convolution1D(
                in_channels=in_channels,
                out_channels=in_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
                groups=in_channels)
            self.pw_conv = conv1d1(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                use_bias=use_bias)
            if self.use_channel_shuffle:
                self.shuffle = ChannelShuffle1d(
                    channels=out_channels,
                    groups=groups)
            if self.use_bn:
                self.bn = L.BatchNormalization(
                    size=out_channels,
                    eps=bn_eps)
            if self.activate:
                self.activ = activation()
            if self.use_dropout:
                self.dropout = partial(
                    F.dropout,
                    ratio=dropout_rate)

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        if self.use_channel_shuffle:
            x = self.shuffle(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


def dwsconv1d1_block(stride=1,
                     pad=0,
                     **kwargs):
    """
    1-dim kernel version of the 1D depthwise version convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default 1.
        Stride of the convolution.
    pad : int, default 0.
        Padding value for convolution layer.
    dilate : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    return DwsConvBlock1d(
        ksize=1,
        stride=stride,
        pad=pad,
        **kwargs)


class QuartzUnit(Chain):
    """
    QuartzNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int
        Convolution window size.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    repeat : int
        Count of body convolution blocks.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 dropout_rate,
                 repeat,
                 **kwargs):
        super(QuartzUnit, self).__init__(**kwargs)
        with self.init_scope():
            self.identity_conv = conv1d1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_rate=0.0,
                activation=None)

            self.body = SimpleSequential()
            with self.body.init_scope():
                for i in range(repeat):
                    activation = (lambda: F.relu) if i < repeat - 1 else None
                    dropout_rate_i = dropout_rate if i < repeat - 1 else 0.0
                    setattr(self.body, "block{}".format(i + 1), DwsConvBlock1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksize=ksize,
                        stride=1,
                        pad=(ksize // 2),
                        dropout_rate=dropout_rate_i,
                        activation=activation))
                    in_channels = out_channels

            self.activ = F.relu
            self.dropout = partial(
                    F.dropout,
                    ratio=dropout_rate)

    def __call__(self, x):
        identity = self.identity_conv(x)
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        x = self.dropout(x)
        return x


class QuartzFinalBlock(Chain):
    """
    QuartzNet specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of output channels for each block.
    ksizes : list of int
        Kernel sizes for each block.
    dropout_rates : list of int
        Dropout rates for each block.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 ksizes,
                 dropout_rates,
                 **kwargs):
        super(QuartzFinalBlock, self).__init__(**kwargs)
        with self.init_scope():
            self.conv1 = DwsConvBlock1d(
                in_channels=in_channels,
                out_channels=channels[-2],
                ksize=ksizes[-2],
                stride=1,
                pad=(2 * ksizes[-2] // 2 - 1),
                dilate=2,
                dropout_rate=dropout_rates[-2])
            self.conv2 = ConvBlock1d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                ksize=ksizes[-1],
                stride=1,
                pad=(ksizes[-1] // 2),
                dropout_rate=dropout_rates[-1])

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class QuartzNet(Chain):
    """
    QuartzNet model from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions,'
    https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit and initial/final block.
    ksizes : list of int
        Kernel sizes for each unit and initial/final block.
    dropout_rates : list of int
        Dropout rates for each unit and initial/final block.
    repeat : int
        Count of body convolution blocks.
    in_channels : int, default 120
        Number of input channels (audio features).
    classes : int, default 11
        Number of classification classes (number of graphemes).
    """
    def __init__(self,
                 channels,
                 ksizes,
                 dropout_rates,
                 repeat,
                 in_channels=120,
                 classes=11,
                 **kwargs):
        super(QuartzNet, self).__init__(**kwargs)
        self.in_size = None
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", DwsConvBlock1d(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    ksize=ksizes[0],
                    stride=2,
                    pad=(ksizes[0] // 2),
                    dropout_rate=dropout_rates[0]))
                in_channels = channels[0]
                for i, (out_channels, ksize, dropout_rate) in\
                        enumerate(zip(channels[1:-2], ksizes[1:-2], dropout_rates[1:-2])):
                    setattr(self.features, "unit{}".format(i + 1), QuartzUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksize=ksize,
                        dropout_rate=dropout_rate,
                        repeat=repeat))
                    in_channels = out_channels
                setattr(self.features, "final_block", QuartzFinalBlock(
                    in_channels=in_channels,
                    channels=channels,
                    ksizes=ksizes,
                    dropout_rates=dropout_rates))
                in_channels = channels[-1]

            self.output = conv1d1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True)

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_quartznet(version,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create QuartzNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    import numpy as np

    blocks, repeat = tuple(map(int, version.split("x")))
    main_stage_repeat = blocks // 5

    channels_per_stage = [256, 256, 256, 512, 512, 512, 512, 1024]
    kernel_sizes_per_stage = [33, 33, 39, 51, 63, 75, 87, 1]
    dropout_rates_per_stage = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    stage_repeat = np.full((8,), 1)
    stage_repeat[1:-2] *= main_stage_repeat
    channels = sum([[a] * r for (a, r) in zip(channels_per_stage, stage_repeat)], [])
    ksizes = sum([[a] * r for (a, r) in zip(kernel_sizes_per_stage, stage_repeat)], [])
    dropout_rates = sum([[a] * r for (a, r) in zip(dropout_rates_per_stage, stage_repeat)], [])

    net = QuartzNet(
        channels=channels,
        ksizes=ksizes,
        dropout_rates=dropout_rates,
        repeat=repeat,
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


def quartznet5x5(**kwargs):
    """
    QuartzNet 5x5 model from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable
    Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(version="5x5", model_name="quartznet5x5", **kwargs)


def quartznet10x5(**kwargs):
    """
    QuartzNet 10x5 model from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable
    Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(version="10x5", model_name="quartznet10x5", **kwargs)


def quartznet15x5(**kwargs):
    """
    QuartzNet 15x5 model from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable
    Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(version="15x5", model_name="quartznet15x5", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False
    audio_features = 120
    classes = 11

    models = [
        quartznet5x5,
        quartznet10x5,
        quartznet15x5,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            classes=classes,
            pretrained=pretrained)

        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != quartznet5x5 or weight_count == 6710915)
        assert (model != quartznet10x5 or weight_count == 12816515)
        assert (model != quartznet15x5 or weight_count == 18922115)

        batch = 1
        seq_len = np.random.randint(60, 150)
        x = np.random.rand(batch, audio_features, seq_len).astype(np.float32)
        y = net(x)
        assert (y.shape[:2] == (batch, classes))
        assert (y.shape[2] in [seq_len // 2, seq_len // 2 + 1])


if __name__ == "__main__":
    _test()
