"""
    Jasper DR (Dense Residual) for ASR, implemented in TensorFlow.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['jasperdr10x5_en', 'jasperdr10x5_en_nr']

from .jasper import get_jasper
from .common import is_channels_first


def jasperdr10x5_en(classes=29, **kwargs):
    """
    Jasper DR 10x5 model for English language from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_jasper(classes=classes, version=("jasper", "10x5"), use_dr=True, model_name="jasperdr10x5_en",
                      **kwargs)


def jasperdr10x5_en_nr(classes=29, **kwargs):
    """
    Jasper DR 10x5 model for English language (with presence of noise) from 'Jasper: An End-to-End Convolutional Neural
    Acoustic Model,' https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_jasper(classes=classes, version=("jasper", "10x5"), use_dr=True, model_name="jasperdr10x5_en_nr",
                      **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K
    import tensorflow as tf

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False
    audio_features = 64
    classes = 29

    models = [
        jasperdr10x5_en,
        jasperdr10x5_en_nr,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            pretrained=pretrained,
            data_format=data_format)

        batch = 3
        seq_len = np.random.randint(60, 150, batch)
        seq_len_max = seq_len.max() + 2
        x = tf.random.normal((batch, audio_features, seq_len_max) if is_channels_first(data_format) else
                             (batch, seq_len_max, audio_features))
        x_len = tf.convert_to_tensor(seq_len.astype(np.long))

        y, y_len = net(x, x_len)
        assert (y.shape.as_list()[0] == batch)
        if is_channels_first(data_format):
            assert (y.shape.as_list()[1] == classes)
            assert (y.shape.as_list()[2] in [seq_len_max // 2, seq_len_max // 2 + 1])
        else:
            assert (y.shape.as_list()[1] in [seq_len_max // 2, seq_len_max // 2 + 1])
            assert (y.shape.as_list()[2] == classes)

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasperdr10x5_en or weight_count == 332632349)
        assert (model != jasperdr10x5_en_nr or weight_count == 332632349)


if __name__ == "__main__":
    _test()
