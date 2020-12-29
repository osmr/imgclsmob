"""
    WRN for CIFAR/SVHN, implemented in TensorFlow.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['CIFARWRN', 'wrn16_10_cifar10', 'wrn16_10_cifar100', 'wrn16_10_svhn', 'wrn28_10_cifar10',
           'wrn28_10_cifar100', 'wrn28_10_svhn', 'wrn40_8_cifar10', 'wrn40_8_cifar100', 'wrn40_8_svhn']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3, SimpleSequential, flatten, is_channels_first
from .preresnet import PreResUnit, PreResActivation


class CIFARWRN(tf.keras.Model):
    """
    WRN model for CIFAR from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 data_format="channels_last",
                 **kwargs):
        super(CIFARWRN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(PreResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bottleneck=False,
                    conv1_stride=False,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(PreResActivation(
            in_channels=in_channels,
            data_format=data_format,
            name="post_activ"))
        self.features.add(nn.AveragePooling2D(
            pool_size=8,
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


def get_wrn_cifar(classes,
                  blocks,
                  width_factor,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
                  **kwargs):
    """
    Create WRN model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    width_factor : int
        Wide scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    assert ((blocks - 4) % 6 == 0)
    layers = [(blocks - 4) // 6] * 3
    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci * width_factor] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFARWRN(
        channels=channels,
        init_block_channels=init_block_channels,
        classes=classes,
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


def wrn16_10_cifar10(classes=10, **kwargs):
    """
    WRN-16-10 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=16, width_factor=10, model_name="wrn16_10_cifar10", **kwargs)


def wrn16_10_cifar100(classes=100, **kwargs):
    """
    WRN-16-10 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=16, width_factor=10, model_name="wrn16_10_cifar100", **kwargs)


def wrn16_10_svhn(classes=10, **kwargs):
    """
    WRN-16-10 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=16, width_factor=10, model_name="wrn16_10_svhn", **kwargs)


def wrn28_10_cifar10(classes=10, **kwargs):
    """
    WRN-28-10 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=28, width_factor=10, model_name="wrn28_10_cifar10", **kwargs)


def wrn28_10_cifar100(classes=100, **kwargs):
    """
    WRN-28-10 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=28, width_factor=10, model_name="wrn28_10_cifar100", **kwargs)


def wrn28_10_svhn(classes=10, **kwargs):
    """
    WRN-28-10 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=28, width_factor=10, model_name="wrn28_10_svhn", **kwargs)


def wrn40_8_cifar10(classes=10, **kwargs):
    """
    WRN-40-8 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=40, width_factor=8, model_name="wrn40_8_cifar10", **kwargs)


def wrn40_8_cifar100(classes=100, **kwargs):
    """
    WRN-40-8 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=40, width_factor=8, model_name="wrn40_8_cifar100", **kwargs)


def wrn40_8_svhn(classes=10, **kwargs):
    """
    WRN-40-8 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=40, width_factor=8, model_name="wrn40_8_svhn", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        (wrn16_10_cifar10, 10),
        (wrn16_10_cifar100, 100),
        (wrn16_10_svhn, 10),
        (wrn28_10_cifar10, 10),
        (wrn28_10_cifar100, 100),
        (wrn28_10_svhn, 10),
        (wrn40_8_cifar10, 10),
        (wrn40_8_cifar100, 100),
        (wrn40_8_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 32, 32) if is_channels_first(data_format) else (batch, 32, 32, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn16_10_cifar10 or weight_count == 17116634)
        assert (model != wrn16_10_cifar100 or weight_count == 17174324)
        assert (model != wrn16_10_svhn or weight_count == 17116634)
        assert (model != wrn28_10_cifar10 or weight_count == 36479194)
        assert (model != wrn28_10_cifar100 or weight_count == 36536884)
        assert (model != wrn28_10_svhn or weight_count == 36479194)
        assert (model != wrn40_8_cifar10 or weight_count == 35748314)
        assert (model != wrn40_8_cifar100 or weight_count == 35794484)
        assert (model != wrn40_8_svhn or weight_count == 35748314)


if __name__ == "__main__":
    _test()
