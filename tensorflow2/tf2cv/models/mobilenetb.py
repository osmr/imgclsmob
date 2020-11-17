"""
    MobileNet(B) with simplified depthwise separable convolution block for ImageNet-1K, implemented in Gluon.
    Original paper: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.
"""

__all__ = ['mobilenetb_w1', 'mobilenetb_w3d4', 'mobilenetb_wd2', 'mobilenetb_wd4']

from .mobilenet import get_mobilenet


def mobilenetb_w1(**kwargs):
    """
    1.0 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=1.0, dws_simplified=True, model_name="mobilenetb_w1", **kwargs)


def mobilenetb_w3d4(**kwargs):
    """
    0.75 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=0.75, dws_simplified=True, model_name="mobilenetb_w3d4", **kwargs)


def mobilenetb_wd2(**kwargs):
    """
    0.5 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=0.5, dws_simplified=True, model_name="mobilenetb_wd2", **kwargs)


def mobilenetb_wd4(**kwargs):
    """
    0.25 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=0.25, dws_simplified=True, model_name="mobilenetb_wd4", **kwargs)


def _test():
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    pretrained = False

    models = [
        mobilenetb_w1,
        mobilenetb_w3d4,
        mobilenetb_wd2,
        mobilenetb_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenetb_w1 or weight_count == 4222056)
        assert (model != mobilenetb_w3d4 or weight_count == 2578120)
        assert (model != mobilenetb_wd2 or weight_count == 1326632)
        assert (model != mobilenetb_wd4 or weight_count == 467592)


if __name__ == "__main__":
    _test()
