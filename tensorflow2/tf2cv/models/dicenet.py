"""
    DiCENet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'DiCENet: Dimension-wise Convolutions for Efficient Networks,' https://arxiv.org/abs/1906.03516.
"""

__all__ = ['DiceNet', 'dicenet_wd5', 'dicenet_wd2', 'dicenet_w3d4', 'dicenet_w1', 'dicenet_w5d4', 'dicenet_w3d2',
           'dicenet_w7d8', 'dicenet_w2']

import os
import math
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, AvgPool2d, MaxPool2d, NormActivation,\
    ChannelShuffle, Concurrent, PReLU2, SimpleSequential, is_channels_first, get_channel_axis, flatten


class SpatialDiceBranch(nn.Layer):
    """
    Spatial element of DiCE block for selected dimension.

    Parameters:
    ----------
    sp_size : int
        Desired size for selected spatial dimension.
    is_height : bool
        Is selected dimension height.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 sp_size,
                 is_height,
                 data_format="channels_last",
                 **kwargs):
        super(SpatialDiceBranch, self).__init__(**kwargs)
        self.is_height = is_height
        self.data_format = data_format
        if is_channels_first(self.data_format):
            self.index = 2 if is_height else 3
        else:
            self.index = 1 if is_height else 2
        self.base_sp_size = sp_size

        self.conv = conv3x3(
            in_channels=self.base_sp_size,
            out_channels=self.base_sp_size,
            groups=self.base_sp_size,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x_shape = x.get_shape().as_list()

        height, width = x_shape[2:4] if is_channels_first(self.data_format) else x_shape[1:3]
        if self.is_height:
            real_sp_size = height
            real_in_size = (real_sp_size, width)
            base_in_size = (self.base_sp_size, width)
        else:
            real_sp_size = width
            real_in_size = (height, real_sp_size)
            base_in_size = (height, self.base_sp_size)

        if real_sp_size != self.base_sp_size:
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=[0, 2, 3, 1])
            x = tf.image.resize(
                images=x,
                size=base_in_size,
                method=self.method)
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=[0, 3, 1, 2])

        if self.is_height:
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=(0, 2, 1, 3))
            else:
                x = tf.transpose(x, perm=(0, 3, 2, 1))
        else:
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=(0, 3, 2, 1))
            else:
                x = tf.transpose(x, perm=(0, 1, 3, 2))

        x = self.conv(x)

        if self.is_height:
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=(0, 2, 1, 3))
            else:
                x = tf.transpose(x, perm=(0, 3, 2, 1))
        else:
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=(0, 3, 2, 1))
            else:
                x = tf.transpose(x, perm=(0, 1, 3, 2))

        changed_sp_size = x.shape[self.index]
        if real_sp_size != changed_sp_size:
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=[0, 2, 3, 1])
            x = tf.image.resize(
                images=x,
                size=real_in_size,
                method=self.method)
            if is_channels_first(self.data_format):
                x = tf.transpose(x, perm=[0, 3, 1, 2])

        return x


class DiceBaseBlock(nn.Layer):
    """
    Base part of DiCE block (without attention).

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(DiceBaseBlock, self).__init__(**kwargs)
        mid_channels = 3 * channels

        self.convs = Concurrent()
        self.convs.add(conv3x3(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            data_format=data_format,
            name="ch_conv"))
        self.convs.add(SpatialDiceBranch(
            sp_size=in_size[0],
            is_height=True,
            data_format=data_format,
            name="h_conv"))
        self.convs.add(SpatialDiceBranch(
            sp_size=in_size[1],
            is_height=False,
            data_format=data_format,
            name="w_conv"))

        self.norm_activ = NormActivation(
            in_channels=mid_channels,
            activation=(lambda: PReLU2(in_channels=mid_channels, name="activ")),
            data_format=data_format,
            name="norm_activ")
        self.shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=3,
            data_format=data_format,
            name="shuffle")
        self.squeeze_conv = conv1x1_block(
            in_channels=mid_channels,
            out_channels=channels,
            groups=channels,
            activation=(lambda: PReLU2(in_channels=channels, name="activ")),
            data_format=data_format,
            name="squeeze_conv")

    def call(self, x, training=None):
        x = self.convs(x)
        x = self.norm_activ(x, training=training)
        x = self.shuffle(x)
        x = self.squeeze_conv(x, training=training)
        return x


class DiceAttBlock(nn.Layer):
    """
    Pure attention part of DiCE block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    reduction : int, default 4
        Squeeze reduction value.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=4,
                 data_format="channels_last",
                 **kwargs):
        super(DiceAttBlock, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = in_channels // reduction

        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_bias=False,
            data_format=data_format,
            name="conv1")
        self.activ = nn.ReLU()
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=False,
            data_format=data_format,
            name="conv2")
        self.sigmoid = tf.nn.sigmoid

    def call(self, x, training=None):
        w = self.pool(x)
        axis = -1 if is_channels_first(self.data_format) else 1
        w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return w


class DiceBlock(nn.Layer):
    """
    DiCE block (volume-wise separable convolutions).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(DiceBlock, self).__init__(**kwargs)
        proj_groups = math.gcd(in_channels, out_channels)

        self.base_block = DiceBaseBlock(
            channels=in_channels,
            in_size=in_size,
            data_format=data_format,
            name="base_block")
        self.att = DiceAttBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="att")
        self.proj_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=proj_groups,
            activation=(lambda: PReLU2(in_channels=out_channels, name="activ")),
            data_format=data_format,
            name="proj_conv")

    def call(self, x, training=None):
        x = self.base_block(x, training=training)
        w = self.att(x, training=training)
        x = self.proj_conv(x, training=training)
        x = x * w
        return x


class StridedDiceLeftBranch(nn.Layer):
    """
    Left branch of the strided DiCE block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 data_format="channels_last",
                 **kwargs):
        super(StridedDiceLeftBranch, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=channels,
            out_channels=channels,
            strides=2,
            groups=channels,
            activation=(lambda: PReLU2(in_channels=channels, name="activ")),
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1_block(
            in_channels=channels,
            out_channels=channels,
            activation=(lambda: PReLU2(in_channels=channels, name="activ")),
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class StridedDiceRightBranch(nn.Layer):
    """
    Right branch of the strided DiCE block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(StridedDiceRightBranch, self).__init__(**kwargs)
        self.pool = AvgPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")
        self.dice = DiceBlock(
            in_channels=channels,
            out_channels=channels,
            in_size=(in_size[0] // 2, in_size[1] // 2),
            data_format=data_format,
            name="dice")
        self.conv = conv1x1_block(
            in_channels=channels,
            out_channels=channels,
            activation=(lambda: PReLU2(in_channels=channels, name="activ")),
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.pool(x)
        x = self.dice(x, training=training)
        x = self.conv(x, training=training)
        return x


class StridedDiceBlock(nn.Layer):
    """
    Strided DiCE block (strided volume-wise separable convolutions).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(StridedDiceBlock, self).__init__(**kwargs)
        assert (out_channels == 2 * in_channels)

        self.branches = Concurrent()
        self.branches.add(StridedDiceLeftBranch(
            channels=in_channels,
            data_format=data_format,
            name="left_branch"))
        self.branches.add(StridedDiceRightBranch(
            channels=in_channels,
            in_size=in_size,
            data_format=data_format,
            name="right_branch"))
        self.shuffle = ChannelShuffle(
            channels=out_channels,
            groups=2,
            data_format=data_format,
            name="shuffle")

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        x = self.shuffle(x)
        return x


class ShuffledDiceRightBranch(nn.Layer):
    """
    Right branch of the shuffled DiCE block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffledDiceRightBranch, self).__init__(**kwargs)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=(lambda: PReLU2(in_channels=out_channels, name="activ")),
            data_format=data_format,
            name="conv")
        self.dice = DiceBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            in_size=in_size,
            data_format=data_format,
            name="dice")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.dice(x, training=training)
        return x


class ShuffledDiceBlock(nn.Layer):
    """
    Shuffled DiCE block (shuffled volume-wise separable convolutions).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffledDiceBlock, self).__init__(**kwargs)
        self.data_format = data_format
        self.left_part = in_channels - in_channels // 2
        right_in_channels = in_channels - self.left_part
        right_out_channels = out_channels - self.left_part

        self.right_branch = ShuffledDiceRightBranch(
            in_channels=right_in_channels,
            out_channels=right_out_channels,
            in_size=in_size,
            data_format=data_format,
            name="right_branch")
        self.shuffle = ChannelShuffle(
            channels=(2 * right_out_channels),
            groups=2,
            data_format=data_format,
            name="shuffle")

    def call(self, x, training=None):
        axis = get_channel_axis(self.data_format)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=axis)
        x2 = self.right_branch(x2, training=training)
        x = tf.concat([x1, x2], axis=axis)
        x = self.shuffle(x)
        return x


class DiceInitBlock(nn.Layer):
    """
    DiceNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(DiceInitBlock, self).__init__(**kwargs)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            activation=(lambda: PReLU2(in_channels=out_channels, name="activ")),
            data_format=data_format,
            name="conv")
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x


class DiceClassifier(nn.Layer):
    """
    DiceNet specific classifier block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    classes : int, default 1000
        Number of classification classes.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 classes,
                 dropout_rate,
                 data_format="channels_last",
                 **kwargs):
        super(DiceClassifier, self).__init__(**kwargs)
        self.data_format = data_format

        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=4,
            data_format=data_format,
            name="conv1")
        self.dropout = nn.Dropout(
            rate=dropout_rate,
            name="dropout")
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=classes,
            use_bias=True,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        axis = -1 if is_channels_first(self.data_format) else 1
        x = tf.expand_dims(tf.expand_dims(x, axis=axis), axis=axis)

        x = self.conv1(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        return x


class DiceNet(tf.keras.Model):
    """
    DiCENet model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,' https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    dropout_rate : float
        Parameter of Dropout layer in classifier. Faction of the input units to drop.
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
                 classifier_mid_channels,
                 dropout_rate,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DiceNet, self).__init__(**kwargs)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(DiceInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        in_size = (in_size[0] // 4, in_size[1] // 4)
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                unit_class = StridedDiceBlock if j == 0 else ShuffledDiceBlock
                stage.add(unit_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    in_size=in_size,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
                in_size = (in_size[0] // 2, in_size[1] // 2) if j == 0 else in_size
            self.features.add(stage)
        self.features.add(nn.GlobalAvgPool2D(
            data_format=data_format,
            name="final_pool"))

        self.output1 = DiceClassifier(
            in_channels=in_channels,
            mid_channels=classifier_mid_channels,
            classes=classes,
            dropout_rate=dropout_rate,
            data_format=data_format,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x, training=training)
        x = flatten(x, self.data_format)
        return x


def get_dicenet(width_scale,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create DiCENet model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    channels_per_layers_dict = {
        0.2: [32, 64, 128],
        0.5: [48, 96, 192],
        0.75: [86, 172, 344],
        1.0: [116, 232, 464],
        1.25: [144, 288, 576],
        1.5: [176, 352, 704],
        1.75: [210, 420, 840],
        2.0: [244, 488, 976],
        2.4: [278, 556, 1112],
    }

    if width_scale not in channels_per_layers_dict.keys():
        raise ValueError("Unsupported DiceNet with width scale: {}".format(width_scale))

    channels_per_layers = channels_per_layers_dict[width_scale]
    layers = [3, 7, 3]

    if width_scale > 0.2:
        init_block_channels = 24
    else:
        init_block_channels = 16

    channels = [[ci] * li for i, (ci, li) in enumerate(zip(channels_per_layers, layers))]
    for i in range(len(channels)):
        pred_channels = channels[i - 1][-1] if i != 0 else init_block_channels
        channels[i] = [pred_channels * 2] + channels[i]

    if width_scale > 2.0:
        classifier_mid_channels = 1280
    else:
        classifier_mid_channels = 1024

    if width_scale > 1.0:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.1

    net = DiceNet(
        channels=channels,
        init_block_channels=init_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        dropout_rate=dropout_rate,
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


def dicenet_wd5(**kwargs):
    """
    DiCENet x0.2 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=0.2, model_name="dicenet_wd5", **kwargs)


def dicenet_wd2(**kwargs):
    """
    DiCENet x0.5 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=0.5, model_name="dicenet_wd2", **kwargs)


def dicenet_w3d4(**kwargs):
    """
    DiCENet x0.75 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=0.75, model_name="dicenet_w3d4", **kwargs)


def dicenet_w1(**kwargs):
    """
    DiCENet x1.0 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.0, model_name="dicenet_w1", **kwargs)


def dicenet_w5d4(**kwargs):
    """
    DiCENet x1.25 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.25, model_name="dicenet_w5d4", **kwargs)


def dicenet_w3d2(**kwargs):
    """
    DiCENet x1.5 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.5, model_name="dicenet_w3d2", **kwargs)


def dicenet_w7d8(**kwargs):
    """
    DiCENet x1.75 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.75, model_name="dicenet_w7d8", **kwargs)


def dicenet_w2(**kwargs):
    """
    DiCENet x2.0 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=2.0, model_name="dicenet_w2", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        dicenet_wd5,
        dicenet_wd2,
        dicenet_w3d4,
        dicenet_w1,
        dicenet_w5d4,
        dicenet_w3d2,
        dicenet_w7d8,
        dicenet_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dicenet_wd5 or weight_count == 1130704)
        assert (model != dicenet_wd2 or weight_count == 1214120)
        assert (model != dicenet_w3d4 or weight_count == 1495676)
        assert (model != dicenet_w1 or weight_count == 1805604)
        assert (model != dicenet_w5d4 or weight_count == 2162888)
        assert (model != dicenet_w3d2 or weight_count == 2652200)
        assert (model != dicenet_w7d8 or weight_count == 3264932)
        assert (model != dicenet_w2 or weight_count == 3979044)


if __name__ == "__main__":
    _test()
