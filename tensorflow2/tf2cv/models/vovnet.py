"""
    VoVNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.
"""

__all__ = ['VoVNet', 'vovnet27s', 'vovnet39', 'vovnet57']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, SequentialConcurrent, MaxPool2d, SimpleSequential, flatten,\
    is_channels_first


class VoVUnit(nn.Layer):
    """
    VoVNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    branch_channels : int
        Number of output channels for each branch.
    num_branches : int
        Number of branches.
    resize : bool
        Whether to use resize block.
    use_residual : bool
        Whether to use residual block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 branch_channels,
                 num_branches,
                 resize,
                 use_residual,
                 data_format="channels_last",
                 **kwargs):
        super(VoVUnit, self).__init__(**kwargs)
        self.resize = resize
        self.use_residual = use_residual

        if self.resize:
            self.pool = MaxPool2d(
                pool_size=3,
                strides=2,
                ceil_mode=True,
                data_format=data_format,
                name="pool")

        self.branches = SequentialConcurrent(
            data_format=data_format,
            name="branches")
        branch_in_channels = in_channels
        for i in range(num_branches):
            self.branches.add(conv3x3_block(
                in_channels=branch_in_channels,
                out_channels=branch_channels,
                data_format=data_format,
                name="branch{}".format(i + 1)))
            branch_in_channels = branch_channels

        self.concat_conv = conv1x1_block(
            in_channels=(in_channels + num_branches * branch_channels),
            out_channels=out_channels,
            data_format=data_format,
            name="concat_conv")

    def call(self, x, training=None):
        if self.resize:
            x = self.pool(x)
        if self.use_residual:
            identity = x
        x = self.branches(x, training=training)
        x = self.concat_conv(x, training=training)
        if self.use_residual:
            x = x + identity
        return x


class VoVInitBlock(nn.Layer):
    """
    VoVNet specific initial block.

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
        super(VoVInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=2,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class VoVNet(tf.keras.Model):
    """
    VoVNet model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    branch_channels : list of list of int
        Number of branch output channels for each unit.
    num_branches : int
        Number of branches for the each unit.
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
                 branch_channels,
                 num_branches,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(VoVNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        init_block_channels = 128

        self.features = SimpleSequential(name="features")
        self.features.add(VoVInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                use_residual = (j != 0)
                resize = (j == 0) and (i != 0)
                stage.add(VoVUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    branch_channels=branch_channels[i][j],
                    num_branches=num_branches,
                    resize=resize,
                    use_residual=use_residual,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_vovnet(blocks,
               slim=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".tensorflow", "models"),
               **kwargs):
    """
    Create ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    slim : bool, default False
        Whether to use a slim model.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if blocks == 27:
        layers = [1, 1, 1, 1]
    elif blocks == 39:
        layers = [1, 1, 2, 2]
    elif blocks == 57:
        layers = [1, 1, 4, 3]
    else:
        raise ValueError("Unsupported VoVNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 6 + 3 == blocks)

    num_branches = 5
    channels_per_layers = [256, 512, 768, 1024]
    branch_channels_per_layers = [128, 160, 192, 224]
    if slim:
        channels_per_layers = [ci // 2 for ci in channels_per_layers]
        branch_channels_per_layers = [ci // 2 for ci in branch_channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    branch_channels = [[ci] * li for (ci, li) in zip(branch_channels_per_layers, layers)]

    net = VoVNet(
        channels=channels,
        branch_channels=branch_channels,
        num_branches=num_branches,
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


def vovnet27s(**kwargs):
    """
    VoVNet-27-slim model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=27, slim=True, model_name="vovnet27s", **kwargs)


def vovnet39(**kwargs):
    """
    VoVNet-39 model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=39, model_name="vovnet39", **kwargs)


def vovnet57(**kwargs):
    """
    VoVNet-57 model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=57, model_name="vovnet57", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        vovnet27s,
        vovnet39,
        vovnet57,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != vovnet27s or weight_count == 3525736)
        assert (model != vovnet39 or weight_count == 22600296)
        assert (model != vovnet57 or weight_count == 36640296)


if __name__ == "__main__":
    _test()
