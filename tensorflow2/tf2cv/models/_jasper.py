"""
    Jasper for ASR, implemented in TensorFlow.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['Jasper', 'jasper5x3', 'jasper10x4', 'jasper10x5', 'conv1d1', 'ConvBlock1d', 'conv1d1_block',
           'JasperFinalBlock']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import get_activation_layer, Conv1d, BatchNorm, SimpleSequential, is_channels_first


def conv1d1(in_channels,
            out_channels,
            strides=1,
            groups=1,
            use_bias=False,
            data_format="channels_last",
            **kwargs):
    """
    1-dim kernel version of the 1D convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        **kwargs)


class ConvBlock1d(nn.Layer):
    """
    Standard 1D convolution block with batch normalization, activation, and dropout.

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
    force_same : bool, default False
        Whether to forcibly set `same` padding in convolution.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 groups=1,
                 use_bias=False,
                 force_same=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 dropout_rate=0.0,
                 data_format="channels_last",
                 **kwargs):
        super(ConvBlock1d, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_dropout = (dropout_rate != 0.0)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            force_same=force_same,
            data_format=data_format,
            name="conv")
        if self.use_bn:
            self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation, name="activ")
        if self.use_dropout:
            self.dropout = nn.Dropout(
                rate=dropout_rate,
                name="dropout")

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x, training=training)
        return x


def conv1d1_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=0,
                  data_format="channels_last",
                  **kwargs):
    """
    1-dim kernel version of the standard 1D convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int, default 1
        Strides of the convolution.
    padding : int, default 0
        Padding value for convolution layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return ConvBlock1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=padding,
        data_format=data_format,
        **kwargs)


class JasperUnit(nn.Layer):
    """
    Jasper unit with residual connection.

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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout_rate,
                 repeat,
                 data_format="channels_last",
                 **kwargs):
        super(JasperUnit, self).__init__(**kwargs)
        self.identity_conv = conv1d1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_rate=0.0,
            activation=None,
            data_format=data_format,
            name="identity_conv")

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

    def call(self, x, training=None):
        identity = self.identity_conv(x, training=training)
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        x = self.dropout(x, training=training)
        return x


class JasperFinalBlock(nn.Layer):
    """
    Jasper specific final block.

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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_sizes,
                 dropout_rates,
                 data_format="channels_last",
                 **kwargs):
        super(JasperFinalBlock, self).__init__(**kwargs)
        self.conv1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=channels[-2],
            kernel_size=kernel_sizes[-2],
            strides=1,
            padding=(2 * kernel_sizes[-2] // 2 - 1),
            dilation=2,
            dropout_rate=dropout_rates[-2],
            data_format=data_format,
            name="conv1")
        self.conv2 = ConvBlock1d(
            in_channels=channels[-2],
            out_channels=channels[-1],
            kernel_size=kernel_sizes[-1],
            strides=1,
            padding=(kernel_sizes[-1] // 2),
            dropout_rate=dropout_rates[-1],
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class Jasper(tf.keras.Model):
    """
    Jasper model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.

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
        super(Jasper, self).__init__(**kwargs)
        self.in_size = None
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
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
        for i, (out_channels, kernel_size, dropout_rate) in\
                enumerate(zip(channels[1:-2], kernel_sizes[1:-2], dropout_rates[1:-2])):
            self.features.add(JasperUnit(
                in_channels=in_channels,
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


def get_jasper(version,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
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

    net = Jasper(
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


def jasper5x3(**kwargs):
    """
    Jasper 5x3 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version="10x5", model_name="jasper10x5", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
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
        assert (model != jasper5x3 or weight_count == 107820299)
        assert (model != jasper10x4 or weight_count == 261532939)
        assert (model != jasper10x5 or weight_count == 322426123)


if __name__ == "__main__":
    _test()
