"""
    QuartzNet for ASR, implemented in Gluon.
    Original paper: 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions,'
    https://arxiv.org/abs/1910.10261.
"""

__all__ = ['QuartzNet', 'quartznet5x5', 'quartznet10x5', 'quartznet15x5']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .jasper import conv1d1, conv1d1_block, ConvBlock1d
from .common import BatchNormExtra, ChannelShuffle


class DwsConvBlock1d(HybridBlock):
    """
    Depthwise version of the 1D standard convolution block with batch normalization, activation, dropout, and channel
    shuffle.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    strides : int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 dropout_rate=0.0,
                 **kwargs):
        super(DwsConvBlock1d, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_dropout = (dropout_rate != 0.0)
        self.use_channel_shuffle = (groups > 1)

        with self.name_scope():
            self.dw_conv = nn.Conv1D(
                channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                use_bias=use_bias,
                in_channels=in_channels)
            self.pw_conv = conv1d1(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                use_bias=use_bias)
            if self.use_channel_shuffle:
                self.shuffle = ChannelShuffle(
                    channels=out_channels,
                    groups=groups)
            if self.use_bn:
                self.bn = BatchNormExtra(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats,
                    cudnn_off=bn_cudnn_off)
            if self.activate:
                self.activ = activation()
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
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
                     padding=0,
                     **kwargs):
    """
    1-dim kernel version of the 1D depthwise version convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    strides : int, default 1
        Strides of the convolution.
    padding : int, default 0
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    return DwsConvBlock1d(
        kernel_size=1,
        stride=stride,
        padding=padding,
        **kwargs)


class QuartzUnit(HybridBlock):
    """
    QuartzNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    repeat : int
        Count of body convolution blocks.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout_rate,
                 repeat,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(QuartzUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.identity_conv = conv1d1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_rate=0.0,
                activation=None,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

            self.body = nn.HybridSequential(prefix="")
            for i in range(repeat):
                activation = (lambda: nn.Activation("relu")) if i < repeat - 1 else None
                dropout_rate_i = dropout_rate if i < repeat - 1 else 0.0
                self.body.add(DwsConvBlock1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=1,
                    padding=(kernel_size // 2),
                    dropout_rate=dropout_rate_i,
                    activation=activation,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
                in_channels = out_channels

            self.activ = nn.Activation("relu")
            self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        identity = self.identity_conv(x)
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        x = self.dropout(x)
        return x


class QuartzFinalBlock(HybridBlock):
    """
    QuartzNet specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of output channels for each block.
    kernel_sizes : list of int
        Kernel sizes for each block.
    dropout_rates : list of int
        Dropout rates for each block.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_sizes,
                 dropout_rates,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(QuartzFinalBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = DwsConvBlock1d(
                in_channels=in_channels,
                out_channels=channels[-2],
                kernel_size=kernel_sizes[-2],
                strides=1,
                padding=(2 * kernel_sizes[-2] // 2 - 1),
                dilation=2,
                dropout_rate=dropout_rates[-2],
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2 = ConvBlock1d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=kernel_sizes[-1],
                strides=1,
                padding=(kernel_sizes[-1] // 2),
                dilation=2,
                dropout_rate=dropout_rates[-1],
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class QuartzNet(HybridBlock):
    """
    QuartzNet model from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions,'
    https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit and initial/final block.
    kernel_sizes : list of int
        Kernel sizes for each unit and initial/final block.
    dropout_rates : list of int
        Dropout rates for each unit and initial/final block.
    repeat : int
        Count of body convolution blocks.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    in_channels : int, default 120
        Number of input channels (audio features).
    classes : int, default 11
        Number of classification classes (number of graphemes).
    """
    def __init__(self,
                 channels,
                 kernel_sizes,
                 dropout_rates,
                 repeat,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_channels=120,
                 classes=11,
                 **kwargs):
        super(QuartzNet, self).__init__(**kwargs)
        self.in_size = None
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(DwsConvBlock1d(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=kernel_sizes[0],
                strides=2,
                padding=(kernel_sizes[0] // 2),
                dropout_rate=dropout_rates[0],
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = channels[0]
            for i, (out_channels, kernel_size, dropout_rate) in\
                    enumerate(zip(channels[1:-2], kernel_sizes[1:-2], dropout_rates[1:-2])):
                self.features.add(QuartzUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    repeat=repeat,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
                in_channels = out_channels
            self.features.add(QuartzFinalBlock(
                in_channels=in_channels,
                channels=channels,
                kernel_sizes=kernel_sizes,
                dropout_rates=dropout_rates,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = channels[-1]

            self.output = conv1d1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_quartznet(version,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    kernel_sizes = sum([[a] * r for (a, r) in zip(kernel_sizes_per_stage, stage_repeat)], [])
    dropout_rates = sum([[a] * r for (a, r) in zip(dropout_rates_per_stage, stage_repeat)], [])

    net = QuartzNet(
        channels=channels,
        kernel_sizes=kernel_sizes,
        dropout_rates=dropout_rates,
        repeat=repeat,
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


def quartznet5x5(**kwargs):
    """
    QuartzNet 5x5 model from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable
    Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(version="15x5", model_name="quartznet15x5", **kwargs)


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
    import numpy as np
    import mxnet as mx

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

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != quartznet5x5 or weight_count == 6710915)
        assert (model != quartznet10x5 or weight_count == 12816515)
        assert (model != quartznet15x5 or weight_count == 18922115)

        batch = 4
        seq_len = np.random.randint(60, 150)
        x = mx.nd.random.normal(shape=(batch, audio_features, seq_len), ctx=ctx)
        y = net(x)
        assert (y.shape[:2] == (batch, classes))
        assert (y.shape[2] in [seq_len // 2, seq_len // 2 + 1])


if __name__ == "__main__":
    _test()
