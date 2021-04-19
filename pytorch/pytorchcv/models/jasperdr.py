"""
    Jasper DR (Dense Residual) for ASR, implemented in PyTorch.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['jasperdr5x3', 'jasperdr10x4', 'jasperdr10x5']

from .jasper import get_jasper


def jasperdr5x3(**kwargs):
    """
    Jasper DR 5x3 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version=("jasper", "5x3"), use_dr=True, model_name="jasperdr5x3", **kwargs)


def jasperdr10x4(**kwargs):
    """
    Jasper DR 10x4 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version=("jasper", "10x4"), use_dr=True, model_name="jasperdr10x4", **kwargs)


def jasperdr10x5(**kwargs):
    """
    Jasper DR 10x5 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version=("jasper", "10x5"), use_dr=True, model_name="jasperdr10x5", **kwargs)


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
    audio_features = 120
    num_classes = 11

    models = [
        jasperdr5x3,
        jasperdr10x4,
        jasperdr10x5,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            num_classes=num_classes,
            pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasperdr5x3 or weight_count == 109848331)
        assert (model != jasperdr10x4 or weight_count == 271878411)
        assert (model != jasperdr10x5 or weight_count == 332771595)

        batch = 1
        seq_len = np.random.randint(60, 150)
        x = torch.randn(batch, audio_features, seq_len)
        x_len = torch.tensor(seq_len - 2, dtype=torch.long, device=x.device).unsqueeze(dim=0)
        y, y_len = net(x, x_len)
        # y.sum().backward()
        assert (tuple(y.size())[:2] == (batch, num_classes))
        assert (y.size()[2] in [seq_len // 2, seq_len // 2 + 1])


if __name__ == "__main__":
    _test()
