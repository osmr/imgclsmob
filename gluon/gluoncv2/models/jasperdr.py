"""
    Jasper DR (Dense Residual) for ASR, implemented in Gluon.
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_jasper(classes=classes, version=("jasper", "10x5"), use_dr=True, model_name="jasperdr10x5_en_nr",
                      **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def _test():
    import numpy as np
    import mxnet as mx

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

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasperdr10x5_en or weight_count == 332632349)
        assert (model != jasperdr10x5_en_nr or weight_count == 332632349)

        batch = 3
        seq_len = np.random.randint(60, 150, batch)
        seq_len_max = seq_len.max() + 2
        x = mx.nd.random.normal(shape=(batch, audio_features, seq_len_max), ctx=ctx)
        x_len = mx.nd.array(seq_len, ctx=ctx, dtype=np.long)

        y, y_len = net(x, x_len)
        assert (y.shape[:2] == (batch, net.classes))
        assert (y.shape[2] in [seq_len_max // 2, seq_len_max // 2 + 1])


if __name__ == "__main__":
    _test()
