"""
    ESPNetv2 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.
    NB: not ready.
"""

__all__ = ['ESPNetv2', 'espnetv2_wd2', 'espnetv2_w1', 'espnetv2_w5d4', 'espnetv2_w3d2', 'espnetv2_w2']

import os
import math
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import BatchNorm, PReLU2, conv3x3, conv1x1_block, conv3x3_block, AvgPool2d, SimpleSequential,\
    DualPathSequential, flatten, is_channels_first, get_channel_axis


class PreActivation(nn.Layer):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 data_format="channels_last",
                 **kwargs):
        super(PreActivation, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.bn = BatchNorm(
            data_format=data_format,
            name="bn")
        self.activ = PReLU2()

    def call(self, x, training=None):
        x = self.bn(x, training=training)
        x = self.activ(x)
        return x


class ShortcutBlock(nn.Layer):
    """
    ESPNetv2 shortcut block.

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
        super(ShortcutBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            activation=(lambda: PReLU2()),
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class HierarchicalConcurrent(SimpleSequential):
    """
    A container for hierarchical concatenation of blocks with parameters.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(HierarchicalConcurrent, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)

    def call(self, x, training=None):
        out = []
        y_prev = None
        for block in self.children:
            y = block(x, training=training)
            print(y.shape)
            if y_prev is not None:
                y = y + y_prev
            out.append(y)
            y_prev = y
        out = tf.concat(out, axis=self.axis)
        return out


class ESPBlock(nn.Layer):
    """
    ESPNetv2 block (so-called EESP block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    dilations : list of int
        Dilation values for branches.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 dilations,
                 data_format="channels_last",
                 **kwargs):
        super(ESPBlock, self).__init__(**kwargs)
        num_branches = len(dilations)
        assert (out_channels % num_branches == 0)
        self.downsample = (strides != 1)
        mid_channels = out_channels // num_branches

        # dilations = [1] * len(dilations)

        self.reduce_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=num_branches,
            activation=(lambda: PReLU2()),
            data_format=data_format,
            name="reduce_conv")

        self.branches = HierarchicalConcurrent(
            data_format=data_format,
            name="branches")
        for i in range(num_branches):
            self.branches.add(conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                padding=dilations[i],
                dilation=dilations[i],
                groups=mid_channels,
                data_format=data_format,
                name="branch{}".format(i + 1)))

        self.merge_conv = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=num_branches,
            activation=None,
            data_format=data_format,
            name="merge_conv")
        self.preactiv = PreActivation(
            in_channels=out_channels,
            data_format=data_format,
            name="preactiv")
        if not self.downsample:
            self.activ = PReLU2()

    def call(self, x, x0, training=None):
        y = self.reduce_conv(x, training=training)
        y = self.branches(y, training=training)
        y = self.preactiv(y, training=training)
        y = self.merge_conv(y, training=training)
        if not self.downsample:
            y = y + x
            y = self.activ(y)
        return y, x0


class DownsampleBlock(nn.Layer):
    """
    ESPNetv2 downsample block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    x0_channels : int
        Number of input channels for shortcut.
    dilations : list of int
        Dilation values for branches in EESP block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 x0_channels,
                 dilations,
                 data_format="channels_last",
                 **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)
        self.data_format = data_format
        inc_channels = out_channels - in_channels

        # dilations = [1] * len(dilations)

        self.pool = AvgPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            ceil_mode=True,
            data_format=data_format,
            name="pool")
        self.eesp = ESPBlock(
            in_channels=in_channels,
            out_channels=inc_channels,
            strides=2,
            dilations=dilations,
            data_format=data_format,
            name="eesp")
        self.shortcut_block = ShortcutBlock(
            in_channels=x0_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="shortcut_block")
        self.activ = PReLU2()

    def call(self, x, x0, training=None):
        y1 = self.pool(x)
        y2, _ = self.eesp(x, None, training=training)
        x = tf.concat([y1, y2], axis=get_channel_axis(self.data_format))
        x0 = self.pool(x0)
        y3 = self.shortcut_block(x0, training=training)
        x = x + y3
        x = self.activ(x)
        return x, x0


class ESPInitBlock(nn.Layer):
    """
    ESPNetv2 initial block.

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
        super(ESPInitBlock, self).__init__(**kwargs)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            activation=(lambda: PReLU2()),
            data_format=data_format,
            name="conv")
        self.pool = AvgPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, x0, training=None):
        x = self.conv(x, training=training)
        x0 = self.pool(x0)
        return x, x0


class ESPFinalBlock(nn.Layer):
    """
    ESPNetv2 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    final_groups : int
        Number of groups in the last convolution layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 final_groups,
                 data_format="channels_last",
                 **kwargs):
        super(ESPFinalBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            activation=(lambda: PReLU2()),
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=final_groups,
            activation=(lambda: PReLU2()),
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class ESPNetv2(tf.keras.Model):
    """
    ESPNetv2 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    final_block_groups : int
        Number of groups for the final unit.
    dilations : list of list of list of int
        Dilation values for branches in each unit.
    dropout_rate : float, default 0.2
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 final_block_channels,
                 final_block_groups,
                 dilations,
                 dropout_rate=0.2,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ESPNetv2, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        x0_channels = in_channels

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=0,
            last_ordinals=2,
            name="features")
        self.features.add(ESPInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential(name="stage{}_".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    unit = DownsampleBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        x0_channels=x0_channels,
                        dilations=dilations[i][j],
                        data_format=data_format,
                        name="unit{}".format(j + 1))
                else:
                    unit = ESPBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=1,
                        dilations=dilations[i][j],
                        data_format=data_format,
                        name="unit{}".format(j + 1))
                stage.add(unit)
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(ESPFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            final_groups=final_block_groups,
            data_format=data_format,
            name="final_block"))
        in_channels = final_block_channels
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(nn.Dropout(
            rate=dropout_rate,
            name="dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="fc"))

    def call(self, x, training=None):
        x = self.features(x, x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_espnetv2(width_scale,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create ESPNetv2 model with specific parameters.

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
    assert (width_scale <= 2.0)

    branches = 4
    layers = [1, 4, 8, 4]

    max_dilation_list = [6, 5, 4, 3, 2]
    max_dilations = [[max_dilation_list[i]] + [max_dilation_list[i + 1]] * (li - 1) for (i, li) in enumerate(layers)]
    dilations = [[sorted([k + 1 if k < dij else 1 for k in range(branches)]) for dij in di] for di in max_dilations]

    base_channels = 32
    weighed_base_channels = math.ceil(float(math.floor(base_channels * width_scale)) / branches) * branches
    channels_per_layers = [weighed_base_channels * pow(2, i + 1) for i in range(len(layers))]

    init_block_channels = base_channels if weighed_base_channels > base_channels else weighed_base_channels
    final_block_channels = 1024 if width_scale <= 1.5 else 1280

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ESPNetv2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        final_block_groups=branches,
        dilations=dilations,
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


def espnetv2_wd2(**kwargs):
    """
    ESPNetv2 x0.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=0.5, model_name="espnetv2_wd2", **kwargs)


def espnetv2_w1(**kwargs):
    """
    ESPNetv2 x1.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.0, model_name="espnetv2_w1", **kwargs)


def espnetv2_w5d4(**kwargs):
    """
    ESPNetv2 x1.25 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.25, model_name="espnetv2_w5d4", **kwargs)


def espnetv2_w3d2(**kwargs):
    """
    ESPNetv2 x1.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.5, model_name="espnetv2_w3d2", **kwargs)


def espnetv2_w2(**kwargs):
    """
    ESPNetv2 x2.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=2.0, model_name="espnetv2_w2", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        espnetv2_wd2,
        espnetv2_w1,
        espnetv2_w5d4,
        espnetv2_w3d2,
        espnetv2_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espnetv2_wd2 or weight_count == 1241092)
        assert (model != espnetv2_w1 or weight_count == 1669592)
        assert (model != espnetv2_w5d4 or weight_count == 1964832)
        assert (model != espnetv2_w3d2 or weight_count == 2314120)
        assert (model != espnetv2_w2 or weight_count == 3497144)


if __name__ == "__main__":
    _test()
