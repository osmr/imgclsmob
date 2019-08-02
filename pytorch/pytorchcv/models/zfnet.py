"""
    ZFNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.
"""

__all__ = ['zfnet', 'zfnetb']

import os
from .alexnet import AlexNet


def get_zfnet(version="a",
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
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
    root : str, default '~/.torch/models'
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

    net = AlexNet(
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_zfnet(version="b", model_name="zfnetb", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        zfnet,
        zfnetb,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != zfnet or weight_count == 62357608)
        assert (model != zfnetb or weight_count == 107627624)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
