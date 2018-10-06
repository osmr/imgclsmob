# from .models.shufflenet import *
from .models.shufflenetv2 import *

__all__ = ['get_model']


_models = {
    'shufflenetv2_wd2': shufflenetv2_wd2,
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    HybridBlock
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError('Unsupported model: {}'.format(name))
    net = _models[name](**kwargs)
    return net
