"""
    InceptionResNetV1 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV1', 'inceptionresnetv1', 'InceptionAUnit', 'InceptionBUnit', 'InceptionCUnit',
           'ReductionAUnit', 'ReductionBUnit']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, BatchNorm, conv1x1, conv1x1_block, conv3x3_block, Concurrent, flatten,\
    is_channels_first, SimpleSequential
from .inceptionv3 import MaxPoolBranch, Conv1x1Branch, ConvSeqBranch


class InceptionAUnit(nn.Layer):
    """
    InceptionResNetV1 type Inception-A unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionAUnit, self).__init__(**kwargs)
        self.scale = 0.17

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:3],
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[3:6],
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch3"))
        conv_in_channels = out_channels_list[0] + out_channels_list[2] + out_channels_list[5]
        self.conv = conv1x1(
            in_channels=conv_in_channels,
            out_channels=in_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        identity = x
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class InceptionBUnit(nn.Layer):
    """
    InceptionResNetV1 type Inception-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionBUnit, self).__init__(**kwargs)
        self.scale = 0.10

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:4],
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        conv_in_channels = out_channels_list[0] + out_channels_list[3]
        self.conv = conv1x1(
            in_channels=conv_in_channels,
            out_channels=in_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        identity = x
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class InceptionCUnit(nn.Layer):
    """
    InceptionResNetV1 type Inception-C unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    scale : float, default 1.0
        Scale value for residual branch.
    activate : bool, default True
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps,
                 scale=0.2,
                 activate=True,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionCUnit, self).__init__(**kwargs)
        self.activate = activate
        self.scale = scale

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Conv1x1Branch(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:4],
            kernel_size_list=(1, (1, 3), (3, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 1), (1, 0)),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        conv_in_channels = out_channels_list[0] + out_channels_list[3]
        self.conv = conv1x1(
            in_channels=conv_in_channels,
            out_channels=in_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        if self.activate:
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        identity = x
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        x = self.scale * x + identity
        if self.activate:
            x = self.activ(x)
        return x


class ReductionAUnit(nn.Layer):
    """
    InceptionResNetV1 type Reduction-A unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionAUnit, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[0:1],
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:4],
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(MaxPoolBranch(
            data_format=data_format,
            name="branch3"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class ReductionBUnit(nn.Layer):
    """
    InceptionResNetV1 type Reduction-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionBUnit, self).__init__(**kwargs)
        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[0:2],
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[2:4],
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[4:7],
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            bn_eps=bn_eps,
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(MaxPoolBranch(
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class InceptInitBlock(nn.Layer):
    """
    InceptionResNetV1 specific initial block.

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
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            data_format=data_format,
            name="pool")
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
        self.conv6 = conv3x3_block(
            in_channels=192,
            out_channels=256,
            strides=2,
            padding=0,
            bn_eps=bn_eps,
            data_format=data_format,
            name="conv6")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool(x)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        return x


class InceptHead(nn.Layer):
    """
    InceptionResNetV1 specific classification block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    dropout_rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 bn_eps,
                 dropout_rate,
                 classes,
                 data_format="channels_last",
                 **kwargs):
        super(InceptHead, self).__init__(**kwargs)
        self.data_format = data_format
        self.use_dropout = (dropout_rate != 0.0)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(
                rate=dropout_rate,
                name="dropout")
        self.fc1 = nn.Dense(
            units=512,
            input_dim=in_channels,
            use_bias=False,
            name="fc1")
        self.bn = BatchNorm(
            epsilon=bn_eps,
            data_format=data_format,
            name="bn")
        self.fc2 = nn.Dense(
            units=classes,
            input_dim=512,
            name="fc2")

    def call(self, x, training=None):
        x = flatten(x, self.data_format)
        if self.use_dropout:
            x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.bn(x, training=training)
        x = self.fc2(x)
        return x


class InceptionResNetV1(tf.keras.Model):
    """
    InceptionResNetV1 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
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
        super(InceptionResNetV1, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        layers = [5, 11, 7]
        in_channels_list = [256, 896, 1792]
        normal_out_channels_list = [[32, 32, 32, 32, 32, 32], [128, 128, 128, 128], [192, 192, 192, 192]]
        reduction_out_channels_list = [[384, 192, 192, 256], [256, 384, 256, 256, 256, 256, 256]]

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
        self.features.add(nn.AveragePooling2D(
            pool_size=8,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = InceptHead(
            in_channels=in_channels,
            bn_eps=bn_eps,
            dropout_rate=dropout_rate,
            classes=classes,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x, training=training)
        return x


def get_inceptionresnetv1(model_name=None,
                          pretrained=False,
                          root=os.path.join("~", ".tensorflow", "models"),
                          **kwargs):
    """
    Create InceptionResNetV1 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = InceptionResNetV1(**kwargs)

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


def inceptionresnetv1(**kwargs):
    """
    InceptionResNetV1 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_inceptionresnetv1(model_name="inceptionresnetv1", bn_eps=1e-3, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        inceptionresnetv1,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 299, 299) if is_channels_first(data_format) else (batch, 299, 299, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionresnetv1 or weight_count == 23995624)


if __name__ == "__main__":
    _test()
