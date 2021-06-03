"""
    Jasper DR (Dense Residual) for ASR, implemented in Chainer.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['jasperdr10x5_en', 'jasperdr10x5_en_nr']

from .jasper import get_jasper


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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_jasper(classes=classes, version=("jasper", "10x5"), use_dr=True, model_name="jasperdr10x5_en_nr",
                      **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False
    audio_features = 64

    models = [
        jasperdr10x5_en,
        jasperdr10x5_en_nr,
    ]

    for model in models:
        net = model(
            in_channels=audio_features,
            pretrained=pretrained)

        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasperdr10x5_en or weight_count == 332632349)
        assert (model != jasperdr10x5_en_nr or weight_count == 332632349)

        batch = 3
        seq_len = np.random.randint(60, 150, batch)
        seq_len_max = seq_len.max() + 2
        x = np.random.rand(batch, audio_features, seq_len_max).astype(np.float32)
        x_len = seq_len.astype(np.long)

        y, y_len = net(x, x_len)
        assert (y.shape[:2] == (batch, net.classes))
        assert (y.shape[2] in [seq_len_max // 2, seq_len_max // 2 + 1])


if __name__ == "__main__":
    _test()
