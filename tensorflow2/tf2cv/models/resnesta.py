"""
    ResNeSt(A) with average downsampling for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.
"""

__all__ = ['ResNeStA', 'resnestabc14', 'resnesta18', 'resnestabc26', 'resnesta50', 'resnesta101', 'resnesta152',
           'resnesta200', 'resnesta269', 'ResNeStADownBlock']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, saconv3x3_block, AvgPool2d, SimpleSequential, is_channels_first
from .senet import SEInitBlock


class ResNeStABlock(nn.Layer):
    """
    Simple ResNeSt(A) block for residual path in ResNeSt(A) unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_bias=False,
                 use_bn=True,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeStABlock, self).__init__(**kwargs)
        self.resize = (strides > 1)

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv1")
        if self.resize:
            self.pool = AvgPool2d(
                pool_size=3,
                strides=strides,
                padding=1,
                data_format=data_format,
                name="pool")
        self.conv2 = saconv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            activation=None,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        if self.resize:
            x = self.pool(x)
        x = self.conv2(x, training=training)
        return x


class ResNeStABottleneck(nn.Layer):
    """
    ResNeSt(A) bottleneck block for residual path in ResNeSt(A) unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck_factor=4,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeStABottleneck, self).__init__(**kwargs)
        self.resize = (strides > 1)
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = saconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv2")
        if self.resize:
            self.pool = AvgPool2d(
                pool_size=3,
                strides=strides,
                padding=1,
                data_format=data_format,
                name="pool")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        if self.resize:
            x = self.pool(x)
        x = self.conv3(x, training=training)
        return x


class ResNeStADownBlock(nn.Layer):
    """
    ResNeSt(A) downsample block for the identity branch of a residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeStADownBlock, self).__init__(**kwargs)
        self.pool = AvgPool2d(
            pool_size=strides,
            strides=strides,
            ceil_mode=True,
            data_format=data_format,
            name="pool")
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.pool(x)
        x = self.conv(x, training=training)
        return x


class ResNeStAUnit(nn.Layer):
    """
    ResNeSt(A) unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck=True,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeStAUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        if bottleneck:
            self.body = ResNeStABottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                data_format=data_format,
                name="body")
        else:
            self.body = ResNeStABlock(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                data_format=data_format,
                name="body")
        if self.resize_identity:
            self.identity_block = ResNeStADownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                data_format=data_format,
                name="identity_block")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_block(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class ResNeStA(tf.keras.Model):
    """
    ResNeSt(A) with average downsampling model from 'ResNeSt: Split-Attention Networks,'
    https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ResNeStA, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(SEInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(ResNeStAUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bottleneck=bottleneck,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.GlobalAvgPool2D(
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        if dropout_rate > 0.0:
            self.output1.add(nn.Dropout(
                rate=dropout_rate,
                name="output1/dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1/fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        return x


def get_resnesta(blocks,
                 bottleneck=None,
                 width_scale=1.0,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create ResNeSt(A) with average downsampling model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    elif blocks == 269:
        layers = [3, 30, 48, 8]
    else:
        raise ValueError("Unsupported ResNeSt(A) with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if blocks >= 101:
        init_block_channels *= 2

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = ResNeStA(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
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


def resnestabc14(**kwargs):
    """
    ResNeSt(A)-BC-14 with average downsampling model from 'ResNeSt: Split-Attention Networks,'
    https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=14, bottleneck=True, model_name="resnestabc14", **kwargs)


def resnesta18(**kwargs):
    """
    ResNeSt(A)-18 with average downsampling model from 'ResNeSt: Split-Attention Networks,'
    https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=18, model_name="resnesta18", **kwargs)


def resnestabc26(**kwargs):
    """
    ResNeSt(A)-BC-26 with average downsampling model from 'ResNeSt: Split-Attention Networks,'
    https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=26, bottleneck=True, model_name="resnestabc26", **kwargs)


def resnesta50(**kwargs):
    """
    ResNeSt(A)-50 with average downsampling model with stride at the second convolution in bottleneck block
    from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=50, model_name="resnesta50", **kwargs)


def resnesta101(**kwargs):
    """
    ResNeSt(A)-101 with average downsampling model with stride at the second convolution in bottleneck
    block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=101, model_name="resnesta101", **kwargs)


def resnesta152(**kwargs):
    """
    ResNeSt(A)-152 with average downsampling model with stride at the second convolution in bottleneck
    block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=152, model_name="resnesta152", **kwargs)


def resnesta200(in_size=(256, 256), **kwargs):
    """
    ResNeSt(A)-200 with average downsampling model with stride at the second convolution in bottleneck
    block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    in_size : tuple of two ints, default (256, 256)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=200, in_size=in_size, dropout_rate=0.2, model_name="resnesta200", **kwargs)


def resnesta269(in_size=(320, 320), **kwargs):
    """
    ResNeSt(A)-269 with average downsampling model with stride at the second convolution in bottleneck
    block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

    Parameters:
    ----------
    in_size : tuple of two ints, default (320, 320)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnesta(blocks=269, in_size=in_size, dropout_rate=0.2, model_name="resnesta269", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        (resnestabc14, 224),
        (resnesta18, 224),
        (resnestabc26, 224),
        (resnesta50, 224),
        (resnesta101, 224),
        (resnesta152, 224),
        (resnesta200, 256),
        (resnesta269, 320),
    ]

    for model, size in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, size, size) if is_channels_first(data_format) else (batch, size, size, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnestabc14 or weight_count == 10611688)
        assert (model != resnesta18 or weight_count == 12763784)
        assert (model != resnestabc26 or weight_count == 17069448)
        assert (model != resnesta50 or weight_count == 27483240)
        assert (model != resnesta101 or weight_count == 48275016)
        assert (model != resnesta152 or weight_count == 65316040)
        assert (model != resnesta200 or weight_count == 70201544)
        assert (model != resnesta269 or weight_count == 110929480)


if __name__ == "__main__":
    _test()
