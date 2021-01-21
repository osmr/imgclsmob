"""
    Jasper DR (Dense Residual) for ASR, implemented in TensorFlow.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['JasperDr', 'jasperdr5x3', 'jasperdr10x4', 'jasperdr10x5']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import SimpleSequential, DualPathSequential, ParallelConcurent, is_channels_first, get_channel_axis
from .jasper import conv1d1, ConvBlock1d, conv1d1_block, JasperFinalBlock


class JasperDrUnit(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels_list,
                 out_channels,
                 kernel_size,
                 dropout_rate,
                 repeat,
                 data_format="channels_last",
                 **kwargs):
        super(JasperDrUnit, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)

        self.identity_convs = ParallelConcurent()
        for i, dense_in_channels_i in enumerate(in_channels_list):
            self.identity_convs.add(conv1d1_block(
                in_channels=dense_in_channels_i,
                out_channels=out_channels,
                dropout_rate=0.0,
                activation=None,
                data_format=data_format,
                name="block{}".format(i + 1)))

        in_channels = in_channels_list[-1]
        self.body = SimpleSequential(name="body")
        for i in range(repeat):
            activation = "relu" if i < repeat - 1 else None
            dropout_rate_i = dropout_rate if i < repeat - 1 else 0.0
            self.body.add(ConvBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=(kernel_size // 2),
                dropout_rate=dropout_rate_i,
                activation=activation,
                data_format=data_format,
                name="block{}".format(i + 1)))
            in_channels = out_channels

        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(
            rate=dropout_rate,
            name="dropout")

    def call(self, x, y=None, training=None):
        y = [x] if y is None else y + [x]
        identity = self.identity_convs(y, training=training)
        identity = tf.stack(identity, axis=self.axis)
        identity = tf.math.reduce_sum(identity, axis=self.axis)
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        x = self.dropout(x, training=training)
        return x, y


class JasperDr(tf.keras.Model):
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
    in_channels : int, default 120
        Number of input channels (audio features).
    classes : int, default 11
        Number of classification classes (number of graphemes).
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 kernel_sizes,
                 dropout_rates,
                 repeat,
                 in_channels=120,
                 classes=11,
                 data_format="channels_last",
                 **kwargs):
        super(JasperDr, self).__init__(**kwargs)
        self.in_size = None
        self.classes = classes
        self.data_format = data_format

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=1,
            name="features")
        self.features.add(ConvBlock1d(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=kernel_sizes[0],
            strides=2,
            padding=(kernel_sizes[0] // 2),
            dropout_rate=dropout_rates[0],
            data_format=data_format,
            name="init_block"))
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
                data_format=data_format,
                name="unit{}".format(i + 1)))
            in_channels = out_channels
        self.features.add(JasperFinalBlock(
            in_channels=in_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            dropout_rates=dropout_rates,
            data_format=data_format,
            name="final_block"))
        in_channels = channels[-1]

        self.output1 = conv1d1(
            in_channels=in_channels,
            out_channels=classes,
            use_bias=True,
            data_format=data_format,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        return x


def get_jasperdr(version,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
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
    root : str, default '~/.tensorflow/models'
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
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def jasperdr5x3(**kwargs):
    """
    Jasper DR 5x3 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_jasperdr(version="10x5", model_name="jasperdr10x5", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
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
            pretrained=pretrained,
            data_format=data_format)

        batch = 14
        seq_len = np.random.randint(60, 150)
        x = tf.random.normal((batch, audio_features, seq_len) if is_channels_first(data_format) else
                             (batch, seq_len, audio_features))
        y = net(x)
        assert (y.shape.as_list()[0] == batch)
        if is_channels_first(data_format):
            assert (y.shape.as_list()[1] == classes)
            assert (y.shape.as_list()[2] in [seq_len // 2, seq_len // 2 + 1])
        else:
            assert (y.shape.as_list()[1] in [seq_len // 2, seq_len // 2 + 1])
            assert (y.shape.as_list()[2] == classes)

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasperdr5x3 or weight_count == 109848331)
        assert (model != jasperdr10x4 or weight_count == 271878411)
        assert (model != jasperdr10x5 or weight_count == 332771595)


if __name__ == "__main__":
    _test()
