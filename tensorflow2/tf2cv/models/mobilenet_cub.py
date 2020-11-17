"""
    MobileNet & FD-MobileNet for CUB-200-2011, implemented in TensorFlow.
    Original papers:
    - 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
       https://arxiv.org/abs/1704.04861.
    - 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,' https://arxiv.org/abs/1802.03750.
"""

__all__ = ['mobilenet_w1_cub', 'mobilenet_w3d4_cub', 'mobilenet_wd2_cub', 'mobilenet_wd4_cub', 'fdmobilenet_w1_cub',
           'fdmobilenet_w3d4_cub', 'fdmobilenet_wd2_cub', 'fdmobilenet_wd4_cub']

from .common import is_channels_first
from .mobilenet import get_mobilenet
from .fdmobilenet import get_fdmobilenet


def mobilenet_w1_cub(classes=200, **kwargs):
    """
    1.0 MobileNet-224 model for CUB-200-2011 from 'MobileNets: Efficient Convolutional Neural Networks for Mobile
    Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(classes=classes, width_scale=1.0, model_name="mobilenet_w1_cub", **kwargs)


def mobilenet_w3d4_cub(classes=200, **kwargs):
    """
    0.75 MobileNet-224 model for CUB-200-2011 from 'MobileNets: Efficient Convolutional Neural Networks for Mobile
    Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(classes=classes, width_scale=0.75, model_name="mobilenet_w3d4_cub", **kwargs)


def mobilenet_wd2_cub(classes=200, **kwargs):
    """
    0.5 MobileNet-224 model for CUB-200-2011 from 'MobileNets: Efficient Convolutional Neural Networks for Mobile
    Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(classes=classes, width_scale=0.5, model_name="mobilenet_wd2_cub", **kwargs)


def mobilenet_wd4_cub(classes=200, **kwargs):
    """
    0.25 MobileNet-224 model for CUB-200-2011 from 'MobileNets: Efficient Convolutional Neural Networks for Mobile
    Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(classes=classes, width_scale=0.25, model_name="mobilenet_wd4_cub", **kwargs)


def fdmobilenet_w1_cub(classes=200, **kwargs):
    """
    FD-MobileNet 1.0x model for CUB-200-2011 from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_fdmobilenet(classes=classes, width_scale=1.0, model_name="fdmobilenet_w1_cub", **kwargs)


def fdmobilenet_w3d4_cub(classes=200, **kwargs):
    """
    FD-MobileNet 0.75x model for CUB-200-2011 from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_fdmobilenet(classes=classes, width_scale=0.75, model_name="fdmobilenet_w3d4_cub", **kwargs)


def fdmobilenet_wd2_cub(classes=200, **kwargs):
    """
    FD-MobileNet 0.5x model for CUB-200-2011 from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_fdmobilenet(classes=classes, width_scale=0.5, model_name="fdmobilenet_wd2_cub", **kwargs)


def fdmobilenet_wd4_cub(classes=200, **kwargs):
    """
    FD-MobileNet 0.25x model for CUB-200-2011 from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_fdmobilenet(classes=classes, width_scale=0.25, model_name="fdmobilenet_wd4_cub", **kwargs)


def _test():
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        mobilenet_w1_cub,
        mobilenet_w3d4_cub,
        mobilenet_wd2_cub,
        mobilenet_wd4_cub,
        fdmobilenet_w1_cub,
        fdmobilenet_w3d4_cub,
        fdmobilenet_wd2_cub,
        fdmobilenet_wd4_cub,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 200))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenet_w1_cub or weight_count == 3411976)
        assert (model != mobilenet_w3d4_cub or weight_count == 1970360)
        assert (model != mobilenet_wd2_cub or weight_count == 921192)
        assert (model != mobilenet_wd4_cub or weight_count == 264472)
        assert (model != fdmobilenet_w1_cub or weight_count == 2081288)
        assert (model != fdmobilenet_w3d4_cub or weight_count == 1218104)
        assert (model != fdmobilenet_wd2_cub or weight_count == 583528)
        assert (model != fdmobilenet_wd4_cub or weight_count == 177560)


if __name__ == "__main__":
    _test()
