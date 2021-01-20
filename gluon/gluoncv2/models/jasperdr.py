"""
    Jasper DR (Dense Residual) for ASR, implemented in Gluon.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['JasperDr', 'jasperdr5x3', 'jasperdr10x4', 'jasperdr10x5']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import DualPathSequential, ParallelConcurent
from .jasper import conv1d1, ConvBlock1d, conv1d1_block, JasperFinalBlock


class JasperDrUnit(HybridBlock):
    """
    Jasper DR unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels (for actual input and each identity connections).
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
                 in_channels_list,
                 out_channels,
                 kernel_size,
                 dropout_rate,
                 repeat,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(JasperDrUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.identity_convs = ParallelConcurent()
            for i, dense_in_channels_i in enumerate(in_channels_list):
                self.identity_convs.add(conv1d1_block(
                    in_channels=dense_in_channels_i,
                    out_channels=out_channels,
                    dropout_rate=0.0,
                    activation=None,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))

            in_channels = in_channels_list[-1]
            self.body = nn.HybridSequential(prefix="")
            for i in range(repeat):
                activation = (lambda: nn.Activation("relu")) if i < repeat - 1 else None
                dropout_rate_i = dropout_rate if i < repeat - 1 else 0.0
                self.body.add(ConvBlock1d(
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

    def hybrid_forward(self, F, x, y=None):
        y = [x] if y is None else y + [x]
        identity = self.identity_convs(y)
        identity = F.stack(*identity, axis=1)
        identity = identity.sum(axis=1)
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        x = self.dropout(x)
        return x, y


class JasperDr(HybridBlock):
    """
    Jasper DR model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.

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
        super(JasperDr, self).__init__(**kwargs)
        self.in_size = None
        self.classes = classes

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=1)
            self.features.add(ConvBlock1d(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=kernel_sizes[0],
                strides=2,
                padding=(kernel_sizes[0] // 2),
                dropout_rate=dropout_rates[0],
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = channels[0]
            in_channels_list = []
            for i, (out_channels, kernel_size, dropout_rate) in\
                    enumerate(zip(channels[1:-2], kernel_sizes[1:-2], dropout_rates[1:-2])):
                in_channels_list += [in_channels]
                self.features.add(JasperDrUnit(
                    in_channels_list=in_channels_list,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    repeat=repeat,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
                in_channels = out_channels
            self.features.add(JasperFinalBlock(
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


def get_jasperdr(version,
                 model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", ""),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    kernel_sizes = sum([[a] * r for (a, r) in zip(kernel_sizes_per_stage, stage_repeat)], [])
    dropout_rates = sum([[a] * r for (a, r) in zip(dropout_rates_per_stage, stage_repeat)], [])

    net = JasperDr(
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


def jasperdr5x3(**kwargs):
    """
    Jasper DR 5x3 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_jasperdr(version="10x5", model_name="jasperdr10x5", **kwargs)


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
        jasperdr5x3,
        jasperdr10x4,
        jasperdr10x5,
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
        assert (model != jasperdr5x3 or weight_count == 109848331)
        assert (model != jasperdr10x4 or weight_count == 271878411)
        assert (model != jasperdr10x5 or weight_count == 332771595)

        batch = 4
        seq_len = np.random.randint(60, 150)
        x = mx.nd.random.normal(shape=(batch, audio_features, seq_len), ctx=ctx)
        y = net(x)
        assert (y.shape[:2] == (batch, classes))
        assert (y.shape[2] in [seq_len // 2, seq_len // 2 + 1])


if __name__ == "__main__":
    _test()
