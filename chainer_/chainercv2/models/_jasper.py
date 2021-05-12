"""
    Jasper for ASR, implemented in Chainer.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['Jasper', 'jasper5x3', 'jasper10x4', 'jasper10x5', 'conv1d1', 'ConvBlock1d', 'conv1d1_block',
           'JasperFinalBlock']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


def conv1d1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            use_bias=False,
            **kwargs):
    """
    1-dim kernel version of the 1D convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default 1
        Stride of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return L.Convolution1D(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        nobias=(not use_bias),
        groups=groups,
        **kwargs)


class ConvBlock1d(Chain):
    """
    Standard 1D convolution block with batch normalization, activation, and dropout.

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
        super(ConvBlock1d, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_dropout = (dropout_rate != 0.0)

        with self.init_scope():
            self.conv = L.Convolution1D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
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
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


def conv1d1_block(in_channels,
                  out_channels,
                  stride=1,
                  pad=0,
                  **kwargs):
    """
    1-dim kernel version of the standard 1D convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default 1
        Stride of the convolution.
    pad : int, default 0
        Padding value for convolution layer.
    """
    return ConvBlock1d(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=pad,
        **kwargs)


class JasperUnit(Chain):
    """
    Jasper unit with residual connection.

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
        super(JasperUnit, self).__init__(**kwargs)
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
                    setattr(self.body, "block{}".format(i + 1), ConvBlock1d(
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


class JasperFinalBlock(Chain):
    """
    Jasper specific final block.

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
        super(JasperFinalBlock, self).__init__(**kwargs)
        with self.init_scope():
            self.conv1 = ConvBlock1d(
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


class Jasper(Chain):
    """
    Jasper model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.

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
        super(Jasper, self).__init__(**kwargs)
        self.in_size = None
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", ConvBlock1d(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    ksize=ksizes[0],
                    stride=2,
                    pad=(ksizes[0] // 2),
                    dropout_rate=dropout_rates[0]))
                in_channels = channels[0]
                for i, (out_channels, ksize, dropout_rate) in\
                        enumerate(zip(channels[1:-2], ksizes[1:-2], dropout_rates[1:-2])):
                    setattr(self.features, "unit{}".format(i + 1), JasperUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksize=ksize,
                        dropout_rate=dropout_rate,
                        repeat=repeat))
                    in_channels = out_channels
                setattr(self.features, "final_block", JasperFinalBlock(
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


def get_jasper(version,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create Jasper model with specific parameters.

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

    channels_per_stage = [256, 256, 384, 512, 640, 768, 896, 1024]
    kernel_sizes_per_stage = [11, 11, 13, 17, 21, 25, 29, 1]
    dropout_rates_per_stage = [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    stage_repeat = np.full((8,), 1)
    stage_repeat[1:-2] *= main_stage_repeat
    channels = sum([[a] * r for (a, r) in zip(channels_per_stage, stage_repeat)], [])
    ksizes = sum([[a] * r for (a, r) in zip(kernel_sizes_per_stage, stage_repeat)], [])
    dropout_rates = sum([[a] * r for (a, r) in zip(dropout_rates_per_stage, stage_repeat)], [])

    net = Jasper(
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


def jasper5x3(**kwargs):
    """
    Jasper 5x3 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version="5x3", model_name="jasper5x3", **kwargs)


def jasper10x4(**kwargs):
    """
    Jasper 10x4 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version="10x4", model_name="jasper10x4", **kwargs)


def jasper10x5(**kwargs):
    """
    Jasper 10x5 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version="10x5", model_name="jasper10x5", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False
    audio_features = 120
    classes = 11

    models = [
        jasper5x3,
        jasper10x4,
        jasper10x5,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            classes=classes,
            pretrained=pretrained)

        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasper5x3 or weight_count == 107820299)
        assert (model != jasper10x4 or weight_count == 261532939)
        assert (model != jasper10x5 or weight_count == 322426123)

        batch = 1
        seq_len = np.random.randint(60, 150)
        x = np.random.rand(batch, audio_features, seq_len).astype(np.float32)
        y = net(x)
        assert (y.shape[:2] == (batch, classes))
        assert (y.shape[2] in [seq_len // 2, seq_len // 2 + 1])


if __name__ == "__main__":
    _test()
