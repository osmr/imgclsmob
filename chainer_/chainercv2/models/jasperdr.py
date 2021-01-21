"""
    Jasper DR (Dense Residual) for ASR, implemented in Chainer.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['JasperDr', 'jasperdr5x3', 'jasperdr10x4', 'jasperdr10x5']

import os
import chainer.functions as F
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import DualPathSequential, ParallelConcurent, SimpleSequential
from .jasper import conv1d1, ConvBlock1d, conv1d1_block, JasperFinalBlock


class JasperDrUnit(Chain):
    """
    Jasper DR unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels (for actual input and each identity connections).
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
                 in_channels_list,
                 out_channels,
                 ksize,
                 dropout_rate,
                 repeat,
                 **kwargs):
        super(JasperDrUnit, self).__init__(**kwargs)
        with self.init_scope():
            self.identity_convs = ParallelConcurent()
            with self.identity_convs.init_scope():
                for i, dense_in_channels_i in enumerate(in_channels_list):
                    setattr(self.identity_convs, "block{}".format(i + 1), conv1d1_block(
                        in_channels=dense_in_channels_i,
                        out_channels=out_channels,
                        dropout_rate=0.0,
                        activation=None))

            in_channels = in_channels_list[-1]
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

    def __call__(self, x, y=None):
        y = [x] if y is None else y + [x]
        identity = self.identity_convs(y)
        identity = F.stack(tuple(identity), axis=1)
        identity = F.sum(identity, axis=1)
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        x = self.dropout(x)
        return x, y


class JasperDr(Chain):
    """
    Jasper DR model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.

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
        super(JasperDr, self).__init__(**kwargs)
        self.in_size = None
        self.classes = classes

        with self.init_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=1)
            with self.features.init_scope():
                setattr(self.features, "init_block", ConvBlock1d(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    ksize=ksizes[0],
                    stride=2,
                    pad=(ksizes[0] // 2),
                    dropout_rate=dropout_rates[0]))
                in_channels = channels[0]
                in_channels_list = []
                for i, (out_channels, ksize, dropout_rate) in\
                        enumerate(zip(channels[1:-2], ksizes[1:-2], dropout_rates[1:-2])):
                    in_channels_list += [in_channels]
                    setattr(self.features, "unit{}".format(i + 1), JasperDrUnit(
                        in_channels_list=in_channels_list,
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


def get_jasperdr(version,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".chainer", "models"),
                 **kwargs):
    """
    Create Jasper DR model with specific parameters.

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

    net = JasperDr(
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


def jasperdr5x3(**kwargs):
    """
    Jasper DR 5x3 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_jasperdr(version="5x3", model_name="jasperdr5x3", **kwargs)


def jasperdr10x4(**kwargs):
    """
    Jasper DR 10x4 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_jasperdr(version="10x4", model_name="jasperdr10x4", **kwargs)


def jasperdr10x5(**kwargs):
    """
    Jasper DR 10x5 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_jasperdr(version="10x5", model_name="jasperdr10x5", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False
    audio_features = 120
    classes = 11

    models = [
        jasperdr5x3,
        jasperdr10x4,
        jasperdr10x5,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            classes=classes,
            pretrained=pretrained)

        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasperdr5x3 or weight_count == 109848331)
        assert (model != jasperdr10x4 or weight_count == 271878411)
        assert (model != jasperdr10x5 or weight_count == 332771595)

        batch = 1
        seq_len = np.random.randint(60, 150)
        x = np.random.rand(batch, audio_features, seq_len).astype(np.float32)
        y = net(x)
        assert (y.shape[:2] == (batch, classes))
        assert (y.shape[2] in [seq_len // 2, seq_len // 2 + 1])


if __name__ == "__main__":
    _test()
