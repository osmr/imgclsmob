"""
    ProxylessNAS for CUB-200-2011, implemented in Chainer.
    Original paper: 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.
"""

__all__ = ['proxylessnas_cpu_cub', 'proxylessnas_gpu_cub', 'proxylessnas_mobile_cub', 'proxylessnas_mobile14_cub']

from .proxylessnas import get_proxylessnas


def proxylessnas_cpu_cub(classes=200, **kwargs):
    """
    ProxylessNAS (CPU) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and
    Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(classes=classes, version="cpu", model_name="proxylessnas_cpu_cub", **kwargs)


def proxylessnas_gpu_cub(classes=200, **kwargs):
    """
    ProxylessNAS (GPU) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and
    Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(classes=classes, version="gpu", model_name="proxylessnas_gpu_cub", **kwargs)


def proxylessnas_mobile_cub(classes=200, **kwargs):
    """
    ProxylessNAS (Mobile) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task
    and Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(classes=classes, version="mobile", model_name="proxylessnas_mobile_cub", **kwargs)


def proxylessnas_mobile14_cub(classes=200, **kwargs):
    """
    ProxylessNAS (Mobile-14) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task
    and Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(classes=classes, version="mobile14", model_name="proxylessnas_mobile14_cub", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        proxylessnas_cpu_cub,
        proxylessnas_gpu_cub,
        proxylessnas_mobile_cub,
        proxylessnas_mobile14_cub,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != proxylessnas_cpu_cub or weight_count == 3215248)
        assert (model != proxylessnas_gpu_cub or weight_count == 5736648)
        assert (model != proxylessnas_mobile_cub or weight_count == 3055712)
        assert (model != proxylessnas_mobile14_cub or weight_count == 5423168)

        x = np.zeros((14, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (14, 200))


if __name__ == "__main__":
    _test()
