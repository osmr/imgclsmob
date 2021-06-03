"""
    Jasper DR (Dense Residual) for ASR, implemented in PyTorch.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['jasperdr10x5_en', 'jasperdr10x5_en_nr']

from .jasper import get_jasper


def jasperdr10x5_en(num_classes=29, **kwargs):
    """
    Jasper DR 10x5 model for English language from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(num_classes=num_classes, version=("jasper", "10x5"), use_dr=True, model_name="jasperdr10x5_en",
                      **kwargs)


def jasperdr10x5_en_nr(num_classes=29, **kwargs):
    """
    Jasper DR 10x5 model for English language (with presence of noise) from 'Jasper: An End-to-End Convolutional Neural
    Acoustic Model,' https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(num_classes=num_classes, version=("jasper", "10x5"), use_dr=True, model_name="jasperdr10x5_en_nr",
                      **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import numpy as np
    import torch

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

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasperdr10x5_en or weight_count == 332632349)
        assert (model != jasperdr10x5_en_nr or weight_count == 332632349)

        batch = 3
        seq_len = np.random.randint(60, 150, batch)
        seq_len_max = seq_len.max() + 2
        x = torch.randn(batch, audio_features, seq_len_max)
        x_len = torch.tensor(seq_len, dtype=torch.long, device=x.device)

        y, y_len = net(x, x_len)
        # y.sum().backward()
        assert (tuple(y.size())[:2] == (batch, net.num_classes))
        assert (y.size()[2] in [seq_len_max // 2, seq_len_max // 2 + 1])


if __name__ == "__main__":
    _test()
