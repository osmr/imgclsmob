"""
    AlexNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.
"""

__all__ = ['AlexNet', 'alexnet', 'alexnetb']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import ConvBlock, MaxPool2d, SimpleSequential, flatten, is_channels_first


class AlexConv(ConvBlock):
    """
    AlexNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    use_lrn : bool
        Whether to use LRN layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_lrn,
                 data_format="channels_last",
                 **kwargs):
        super(AlexConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            use_bn=False,
            data_format=data_format,
            **kwargs)
        self.use_lrn = use_lrn

    def call(self, x, training=None):
        x = super(AlexConv, self).call(x, training=training)
        if self.use_lrn:
            x = tf.nn.lrn(x, bias=2, alpha=1e-4, beta=0.75)
        return x


class AlexDense(nn.Layer):
    """
    AlexNet specific dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(AlexDense, self).__init__(**kwargs)
        self.fc = nn.Dense(
            units=out_channels,
            input_dim=in_channels,
            name="fc")
        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(
            rate=0.5,
            name="dropout")

    def call(self, x, training=None):
        x = self.fc(x)
        x = self.activ(x)
        x = self.dropout(x, training=training)
        return x


class AlexOutputBlock(nn.Layer):
    """
    AlexNet specific output block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 classes,
                 **kwargs):
        super(AlexOutputBlock, self).__init__(**kwargs)
        mid_channels = 4096

        self.fc1 = AlexDense(
            in_channels=in_channels,
            out_channels=mid_channels,
            name="fc1")
        self.fc2 = AlexDense(
            in_channels=mid_channels,
            out_channels=mid_channels,
            name="fc2")
        self.fc3 = nn.Dense(
            units=classes,
            input_dim=mid_channels,
            name="fc3")

    def call(self, x, training=None):
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x)
        return x


class AlexNet(tf.keras.Model):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    kernel_sizes : list of list of int
        Convolution window sizes for each unit.
    strides : list of list of int or tuple/list of 2 int
        Strides of the convolution for each unit.
    paddings : list of list of int or tuple/list of 2 int
        Padding value for convolution layer for each unit.
    use_lrn : bool
        Whether to use LRN layer.
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
                 kernel_sizes,
                 strides,
                 paddings,
                 use_lrn,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        for i, channels_per_stage in enumerate(channels):
            use_lrn_i = use_lrn and (i in [0, 1])
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                stage.add(AlexConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i][j],
                    strides=strides[i][j],
                    padding=paddings[i][j],
                    use_lrn=use_lrn_i,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            stage.add(MaxPool2d(
                pool_size=3,
                strides=2,
                padding=0,
                ceil_mode=True,
                data_format=data_format,
                name="pool{}".format(i + 1)))
            self.features.add(stage)

        in_channels = in_channels * 6 * 6
        self.output1 = AlexOutputBlock(
            in_channels=in_channels,
            classes=classes,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x, training=training)
        return x


def get_alexnet(version="a",
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create AlexNet model with specific parameters.

    Parameters:
    ----------
    version : str, default 'a'
        Version of AlexNet ('a' or 'b').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if version == "a":
        channels = [[96], [256], [384, 384, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[0], [2], [1, 1, 1]]
        use_lrn = True
    elif version == "b":
        channels = [[64], [192], [384, 256, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[2], [2], [1, 1, 1]]
        use_lrn = False
    else:
        raise ValueError("Unsupported AlexNet version {}".format(version))

    net = AlexNet(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        use_lrn=use_lrn,
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


def alexnet(**kwargs):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_alexnet(model_name="alexnet", **kwargs)


def alexnetb(**kwargs):
    """
    AlexNet-b model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997. Non-standard version.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_alexnet(version="b", model_name="alexnetb", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        alexnet,
        alexnetb,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != alexnet or weight_count == 62378344)
        assert (model != alexnetb or weight_count == 61100840)


if __name__ == "__main__":
    _test()
