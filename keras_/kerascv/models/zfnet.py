"""
    ZFNet for ImageNet-1K, implemented in Keras.
    Original paper: 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.
"""

__all__ = ['zfnet', 'zfnetb']

import os
from .common import is_channels_first
from .alexnet import alexnet_model


def get_zfnet(version="a",
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".keras", "models"),
              **kwargs):
    """
    Create ZFNet model with specific parameters.

    Parameters:
    ----------
    version : str, default 'a'
        Version of ZFNet ('a' or 'b').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    if version == "a":
        channels = [[96], [256], [384, 384, 256]]
        kernel_sizes = [[7], [5], [3, 3, 3]]
        strides = [[2], [2], [1, 1, 1]]
        paddings = [[1], [0], [1, 1, 1]]
        use_lrn = True
    elif version == "b":
        channels = [[96], [256], [512, 1024, 512]]
        kernel_sizes = [[7], [5], [3, 3, 3]]
        strides = [[2], [2], [1, 1, 1]]
        paddings = [[1], [0], [1, 1, 1]]
        use_lrn = True
    else:
        raise ValueError("Unsupported ZFNet version {}".format(version))

    net = alexnet_model(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        use_lrn=use_lrn,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def zfnet(**kwargs):
    """
    ZFNet model from 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_zfnet(model_name="zfnet", **kwargs)


def zfnetb(**kwargs):
    """
    ZFNet-b model from 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_zfnet(version="b", model_name="zfnetb", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        zfnet,
        zfnetb,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != zfnet or weight_count == 62357608)
        assert (model != zfnetb or weight_count == 107627624)

        if is_channels_first():
            x = np.zeros((1, 3, 224, 224), np.float32)
        else:
            x = np.zeros((1, 224, 224, 3), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
