"""
    DarkNet for ImageNet-1K, implemented in TensorFlow.
    Original source: 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.
"""

__all__ = ['DarkNet', 'darknet_ref', 'darknet_tiny', 'darknet19']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, conv1x1_block, conv3x3_block, MaxPool2d, SimpleSequential, flatten


def dark_convYxY(in_channels,
                 out_channels,
                 alpha,
                 pointwise,
                 data_format="channels_last",
                 **kwargs):
    """
    DarkNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    alpha : float
        Slope coefficient for Leaky ReLU activation.
    pointwise : bool
        Whether use 1x1 (pointwise) convolution or 3x3 convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    if pointwise:
        return conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=nn.LeakyReLU(alpha=alpha),
            data_format=data_format,
            **kwargs)
    else:
        return conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=nn.LeakyReLU(alpha=alpha),
            data_format=data_format,
            **kwargs)


class DarkNet(tf.keras.Model):
    """
    DarkNet model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    odd_pointwise : bool
        Whether pointwise convolution layer is used for each odd unit.
    avg_pool_size : int
        Window size of the final average pooling.
    cls_activ : bool
        Whether classification convolution layer uses an activation.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
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
                 odd_pointwise,
                 avg_pool_size,
                 cls_activ,
                 alpha=0.1,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DarkNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                stage.add(dark_convYxY(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    alpha=alpha,
                    pointwise=(len(channels_per_stage) > 1) and not (((j + 1) % 2 == 1) ^ odd_pointwise),
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            if i != len(channels) - 1:
                stage.add(MaxPool2d(
                    pool_size=2,
                    strides=2,
                    data_format=data_format,
                    name="pool{}".format(i + 1)))
            self.features.add(stage)

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(Conv2d(
            in_channels=in_channels,
            out_channels=classes,
            kernel_size=1,
            data_format=data_format,
            name="final_conv"))
        if cls_activ:
            self.output1.add(nn.LeakyReLU(alpha=alpha))
        self.output1.add(nn.AveragePooling2D(
            pool_size=avg_pool_size,
            strides=1,
            data_format=data_format,
            name="final_pool"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        x = flatten(x, self.data_format)
        return x


def get_darknet(version,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create DarkNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('ref', 'tiny' or '19').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if version == 'ref':
        channels = [[16], [32], [64], [128], [256], [512], [1024]]
        odd_pointwise = False
        avg_pool_size = 3
        cls_activ = True
    elif version == 'tiny':
        channels = [[16], [32], [16, 128, 16, 128], [32, 256, 32, 256], [64, 512, 64, 512, 128]]
        odd_pointwise = True
        avg_pool_size = 14
        cls_activ = False
    elif version == '19':
        channels = [[32], [64], [128, 64, 128], [256, 128, 256], [512, 256, 512, 256, 512],
                    [1024, 512, 1024, 512, 1024]]
        odd_pointwise = False
        avg_pool_size = 7
        cls_activ = False
    else:
        raise ValueError("Unsupported DarkNet version {}".format(version))

    net = DarkNet(
        channels=channels,
        odd_pointwise=odd_pointwise,
        avg_pool_size=avg_pool_size,
        cls_activ=cls_activ,
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


def darknet_ref(**kwargs):
    """
    DarkNet 'Reference' model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_darknet(version="ref", model_name="darknet_ref", **kwargs)


def darknet_tiny(**kwargs):
    """
    DarkNet Tiny model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_darknet(version="tiny", model_name="darknet_tiny", **kwargs)


def darknet19(**kwargs):
    """
    DarkNet-19 model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_darknet(version="19", model_name="darknet19", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        darknet_ref,
        darknet_tiny,
        darknet19,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darknet_ref or weight_count == 7319416)
        assert (model != darknet_tiny or weight_count == 1042104)
        assert (model != darknet19 or weight_count == 20842376)


if __name__ == "__main__":
    _test()
