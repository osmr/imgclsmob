"""
    InceptionResNetV2 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, conv1x1_block, conv3x3_block, SimpleSequential, Concurrent, flatten, is_channels_first
from .inceptionv3 import AvgPoolBranch, Conv1x1Branch, ConvSeqBranch
from .inceptionresnetv1 import InceptionAUnit, InceptionBUnit, InceptionCUnit, ReductionAUnit, ReductionBUnit


class InceptBlock5b(nn.Layer):
    """
    InceptionResNetV2 type Mixed-5b block.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptBlock5b, self).__init__(**kwargs)
        in_channels = 192

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=96,
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(48, 64),
            kernel_size_list=(1, 5),
            strides_list=(1, 1),
            padding_list=(0, 2),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(AvgPoolBranch(
            in_channels=in_channels,
            out_channels=64,
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptInitBlock(nn.Layer):
    """
    InceptionResNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 bn_eps,
                 in_channels,
                 data_format="channels_last",
                 **kwargs):
        super(InceptInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            strides=2,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            strides=1,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=32,
            out_channels=64,
            strides=1,
            padding=1,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv3")
        self.pool1 = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool1")
        self.conv4 = conv1x1_block(
            in_channels=64,
            out_channels=80,
            strides=1,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv4")
        self.conv5 = conv3x3_block(
            in_channels=80,
            out_channels=192,
            strides=1,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv5")
        self.pool2 = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool2")
        self.block = InceptBlock5b(
            bn_eps=bn_eps,
            data_format=data_format,
            name="block")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool1(x)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.pool2(x)
        x = self.block(x, training=training)
        return x


class InceptionResNetV2(tf.keras.Model):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 dropout_rate=0.0,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionResNetV2, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        layers = [10, 21, 11]
        in_channels_list = [320, 1088, 2080]
        normal_out_channels_list = [[32, 32, 32, 32, 48, 64], [192, 128, 160, 192], [192, 192, 224, 256]]
        reduction_out_channels_list = [[384, 256, 256, 384], [256, 384, 256, 288, 256, 288, 320]]

        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = SimpleSequential(name="features")
        self.features.add(InceptInitBlock(
            in_channels=in_channels,
            bn_eps=bn_eps,
            data_format=data_format,
            name="init_block"))
        in_channels = in_channels_list[0]
        for i, layers_per_stage in enumerate(layers):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j in range(layers_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                    out_channels_list_per_stage = reduction_out_channels_list[i - 1]
                else:
                    unit = normal_units[i]
                    out_channels_list_per_stage = normal_out_channels_list[i]
                if (i == len(layers) - 1) and (j == layers_per_stage - 1):
                    unit_kwargs = {"scale": 1.0, "activate": False}
                else:
                    unit_kwargs = {}
                stage.add(unit(
                    in_channels=in_channels,
                    out_channels_list=out_channels_list_per_stage,
                    bn_eps=bn_eps,
                    data_format=data_format,
                    name="unit{}".format(j + 1),
                    **unit_kwargs))
                if (j == 0) and (i != 0):
                    in_channels = in_channels_list[i]
            self.features.add(stage)
        self.features.add(conv1x1_block(
            in_channels=2080,
            out_channels=1536,
            bn_eps=bn_eps,
            data_format=data_format,
            name="final_block"))
        self.features.add(nn.AveragePooling2D(
            pool_size=8,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        if dropout_rate > 0.0:
            self.output1.add(nn.Dropout(
                rate=dropout_rate,
                name="output1/dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=1536,
            name="output1/fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_inceptionresnetv2(model_name=None,
                          pretrained=False,
                          root=os.path.join("~", ".tensorflow", "models"),
                          **kwargs):
    """
    Create InceptionResNetV2 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = InceptionResNetV2(**kwargs)

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


def inceptionresnetv2(**kwargs):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_inceptionresnetv2(model_name="inceptionresnetv2", bn_eps=1e-3, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        inceptionresnetv2,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 299, 299) if is_channels_first(data_format) else (batch, 299, 299, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionresnetv2 or weight_count == 55843464)


if __name__ == "__main__":
    _test()
