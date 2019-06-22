"""
    SE-ResNet for CUB-200-2011, implemented in PyTorch.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['seresnet10_cub', 'seresnet12_cub', 'seresnet14_cub', 'seresnetbc14b_cub', 'seresnet16_cub',
           'seresnet18_cub', 'seresnet26_cub', 'seresnetbc26b_cub', 'seresnet34_cub', 'seresnetbc38b_cub',
           'seresnet50_cub', 'seresnet50b_cub', 'seresnet101_cub', 'seresnet101b_cub', 'seresnet152_cub',
           'seresnet152b_cub', 'seresnet200_cub', 'seresnet200b_cub']

from .seresnet import get_seresnet


def seresnet10_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-10 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=10, model_name="seresnet10_cub", **kwargs)


def seresnet12_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-12 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=12, model_name="seresnet12_cub", **kwargs)


def seresnet14_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-14 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=14, model_name="seresnet14_cub", **kwargs)


def seresnetbc14b_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-BC-14b model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=14, bottleneck=True, conv1_stride=False,
                        model_name="seresnetbc14b_cub", **kwargs)


def seresnet16_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-16 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=16, model_name="seresnet16_cub", **kwargs)


def seresnet18_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-18 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=18, model_name="seresnet18_cub", **kwargs)


def seresnet26_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-26 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=26, bottleneck=False, model_name="seresnet26_cub", **kwargs)


def seresnetbc26b_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-BC-26b model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=26, bottleneck=True, conv1_stride=False,
                        model_name="seresnetbc26b_cub", **kwargs)


def seresnet34_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-34 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=34, model_name="seresnet34_cub", **kwargs)


def seresnetbc38b_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-BC-38b model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=38, bottleneck=True, conv1_stride=False,
                        model_name="seresnetbc38b_cub", **kwargs)


def seresnet50_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-50 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=50, model_name="seresnet50_cub", **kwargs)


def seresnet50b_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-50 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation Networks,'
    https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=50, conv1_stride=False, model_name="seresnet50b_cub", **kwargs)


def seresnet101_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-101 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=101, model_name="seresnet101_cub", **kwargs)


def seresnet101b_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-101 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=101, conv1_stride=False, model_name="seresnet101b_cub",
                        **kwargs)


def seresnet152_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-152 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=152, model_name="seresnet152_cub", **kwargs)


def seresnet152b_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-152 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=152, conv1_stride=False, model_name="seresnet152b_cub",
                        **kwargs)


def seresnet200_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-200 model for CUB-200-2011 from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=200, model_name="seresnet200_cub", **kwargs)


def seresnet200b_cub(num_classes=200, **kwargs):
    """
    SE-ResNet-200 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507. It's an experimental model.

    Parameters:
    ----------
    num_classes : int, default 200
        Number of classification num_classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(num_classes=num_classes, blocks=200, conv1_stride=False, model_name="seresnet200b_cub",
                        **kwargs)


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
        seresnet10_cub,
        seresnet12_cub,
        seresnet14_cub,
        seresnetbc14b_cub,
        seresnet16_cub,
        seresnet18_cub,
        seresnet26_cub,
        seresnetbc26b_cub,
        seresnet34_cub,
        seresnetbc38b_cub,
        seresnet50_cub,
        seresnet50b_cub,
        seresnet101_cub,
        seresnet101b_cub,
        seresnet152_cub,
        seresnet152b_cub,
        seresnet200_cub,
        seresnet200b_cub,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != seresnet10_cub or weight_count == 5052932)
        assert (model != seresnet12_cub or weight_count == 5127496)
        assert (model != seresnet14_cub or weight_count == 5425104)
        assert (model != seresnetbc14b_cub or weight_count == 9126136)
        assert (model != seresnet16_cub or weight_count == 6614240)
        assert (model != seresnet18_cub or weight_count == 11368192)
        assert (model != seresnet26_cub or weight_count == 17683452)
        assert (model != seresnetbc26b_cub or weight_count == 15756776)
        assert (model != seresnet34_cub or weight_count == 21548468)
        assert (model != seresnetbc38b_cub or weight_count == 22387416)
        assert (model != seresnet50_cub or weight_count == 26448824)
        assert (model != seresnet50b_cub or weight_count == 26448824)
        assert (model != seresnet101_cub or weight_count == 47687672)
        assert (model != seresnet101b_cub or weight_count == 47687672)
        assert (model != seresnet152_cub or weight_count == 65182648)
        assert (model != seresnet152b_cub or weight_count == 65182648)
        assert (model != seresnet200_cub or weight_count == 70196664)
        assert (model != seresnet200b_cub or weight_count == 70196664)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 200))


if __name__ == "__main__":
    _test()
