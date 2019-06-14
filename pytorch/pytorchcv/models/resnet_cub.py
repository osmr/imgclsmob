"""
    ResNet for CUB-200-2011, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['resnet10_cub', 'resnet12_cub', 'resnet14_cub', 'resnetbc14b_cub', 'resnet16_cub', 'resnet18_cub',
           'resnet26_cub', 'resnetbc26b_cub', 'resnet34_cub', 'resnetbc38b_cub', 'resnet50_cub', 'resnet50b_cub',
           'resnet101_cub', 'resnet101b_cub', 'resnet152_cub', 'resnet152b_cub', 'resnet200_cub', 'resnet200b_cub']

from .resnet import get_resnet


def resnet10_cub(num_classes=200, **kwargs):
    """
    ResNet-10 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=10, model_name="resnet10_cub", **kwargs)


def resnet12_cub(num_classes=200, **kwargs):
    """
    ResNet-12 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=12, model_name="resnet12_cub", **kwargs)


def resnet14_cub(num_classes=200, **kwargs):
    """
    ResNet-14 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=14, model_name="resnet14_cub", **kwargs)


def resnetbc14b_cub(num_classes=200, **kwargs):
    """
    ResNet-BC-14b model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=14, bottleneck=True, conv1_stride=False,
                      model_name="resnetbc14b_cub", **kwargs)


def resnet16_cub(num_classes=200, **kwargs):
    """
    ResNet-16 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=16, model_name="resnet16_cub", **kwargs)


def resnet18_cub(num_classes=200, **kwargs):
    """
    ResNet-18 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=18, model_name="resnet18_cub", **kwargs)


def resnet26_cub(num_classes=200, **kwargs):
    """
    ResNet-26 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=26, bottleneck=False, model_name="resnet26_cub", **kwargs)


def resnetbc26b_cub(num_classes=200, **kwargs):
    """
    ResNet-BC-26b model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=26, bottleneck=True, conv1_stride=False,
                      model_name="resnetbc26b_cub", **kwargs)


def resnet34_cub(num_classes=200, **kwargs):
    """
    ResNet-34 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=34, model_name="resnet34_cub", **kwargs)


def resnetbc38b_cub(num_classes=200, **kwargs):
    """
    ResNet-BC-38b model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=38, bottleneck=True, conv1_stride=False,
                      model_name="resnetbc38b_cub", **kwargs)


def resnet50_cub(num_classes=200, **kwargs):
    """
    ResNet-50 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=50, model_name="resnet50_cub", **kwargs)


def resnet50b_cub(num_classes=200, **kwargs):
    """
    ResNet-50 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=50, conv1_stride=False, model_name="resnet50b_cub", **kwargs)


def resnet101_cub(num_classes=200, **kwargs):
    """
    ResNet-101 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=101, model_name="resnet101_cub", **kwargs)


def resnet101b_cub(num_classes=200, **kwargs):
    """
    ResNet-101 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=101, conv1_stride=False, model_name="resnet101b_cub", **kwargs)


def resnet152_cub(num_classes=200, **kwargs):
    """
    ResNet-152 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=152, model_name="resnet152_cub", **kwargs)


def resnet152b_cub(num_classes=200, **kwargs):
    """
    ResNet-152 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=152, conv1_stride=False, model_name="resnet152b_cub", **kwargs)


def resnet200_cub(num_classes=200, **kwargs):
    """
    ResNet-200 model for CUB-200-2011 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=200, model_name="resnet200_cub", **kwargs)


def resnet200b_cub(num_classes=200, **kwargs):
    """
    ResNet-200 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    classes : int, default 200
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(num_classes=num_classes, blocks=200, conv1_stride=False, model_name="resnet200b_cub", **kwargs)


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
        resnet10_cub,
        resnet12_cub,
        resnet14_cub,
        resnetbc14b_cub,
        resnet16_cub,
        resnet18_cub,
        resnet26_cub,
        resnetbc26b_cub,
        resnet34_cub,
        resnetbc38b_cub,
        resnet50_cub,
        resnet50b_cub,
        resnet101_cub,
        resnet101b_cub,
        resnet152_cub,
        resnet152b_cub,
        resnet200_cub,
        resnet200b_cub,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnet10_cub or weight_count == 5008392)
        assert (model != resnet12_cub or weight_count == 5082376)
        assert (model != resnet14_cub or weight_count == 5377800)
        assert (model != resnetbc14b_cub or weight_count == 8425736)
        assert (model != resnet16_cub or weight_count == 6558472)
        assert (model != resnet18_cub or weight_count == 11279112)
        assert (model != resnet26_cub or weight_count == 17549832)
        assert (model != resnetbc26b_cub or weight_count == 14355976)
        assert (model != resnet34_cub or weight_count == 21387272)
        assert (model != resnetbc38b_cub or weight_count == 20286216)
        assert (model != resnet50_cub or weight_count == 23917832)
        assert (model != resnet50b_cub or weight_count == 23917832)
        assert (model != resnet101_cub or weight_count == 42909960)
        assert (model != resnet101b_cub or weight_count == 42909960)
        assert (model != resnet152_cub or weight_count == 58553608)
        assert (model != resnet152b_cub or weight_count == 58553608)
        assert (model != resnet200_cub or weight_count == 63034632)
        assert (model != resnet200b_cub or weight_count == 63034632)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 200))


if __name__ == "__main__":
    _test()
