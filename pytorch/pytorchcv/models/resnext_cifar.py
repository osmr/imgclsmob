"""
    ResNeXt for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
"""

__all__ = ['CIFARResNeXt', 'resnext20_16x4d_cifar10', 'resnext20_16x4d_cifar100', 'resnext20_16x4d_svhn',
           'resnext20_32x2d_cifar10', 'resnext20_32x2d_cifar100', 'resnext20_32x2d_svhn',
           'resnext20_32x4d_cifar10', 'resnext20_32x4d_cifar100', 'resnext20_32x4d_svhn',
           'resnext29_32x4d_cifar10', 'resnext29_32x4d_cifar100', 'resnext29_32x4d_svhn',
           'resnext29_16x64d_cifar10', 'resnext29_16x64d_cifar100', 'resnext29_16x64d_svhn',
           'resnext272_1x64d_cifar10', 'resnext272_1x64d_cifar100', 'resnext272_1x64d_svhn',
           'resnext272_2x32d_cifar10', 'resnext272_2x32d_cifar100', 'resnext272_2x32d_svhn']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3_block
from .resnext import ResNeXtUnit


class CIFARResNeXt(nn.Module):
    """
    ResNeXt model for CIFAR from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    num_classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 cardinality,
                 bottleneck_width,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFARResNeXt, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResNeXtUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_resnext_cifar(num_classes,
                      blocks,
                      cardinality,
                      bottleneck_width,
                      model_name=None,
                      pretrained=False,
                      root=os.path.join("~", ".torch", "models"),
                      **kwargs):
    """
    ResNeXt model for CIFAR with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    assert (blocks - 2) % 9 == 0
    layers = [(blocks - 2) // 9] * 3
    channels_per_layers = [256, 512, 1024]
    init_block_channels = 64

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFARResNeXt(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        num_classes=num_classes,
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


def resnext20_16x4d_cifar10(num_classes=10, **kwargs):
    """
    ResNeXt-20 (16x4d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=16, bottleneck_width=4,
                             model_name="resnext20_16x4d_cifar10", **kwargs)


def resnext20_16x4d_cifar100(num_classes=100, **kwargs):
    """
    ResNeXt-20 (16x4d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=16, bottleneck_width=4,
                             model_name="resnext20_16x4d_cifar100", **kwargs)


def resnext20_16x4d_svhn(num_classes=10, **kwargs):
    """
    ResNeXt-20 (16x4d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=16, bottleneck_width=4,
                             model_name="resnext20_16x4d_svhn", **kwargs)


def resnext20_32x2d_cifar10(num_classes=10, **kwargs):
    """
    ResNeXt-20 (32x2d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=32, bottleneck_width=2,
                             model_name="resnext20_32x2d_cifar10", **kwargs)


def resnext20_32x2d_cifar100(num_classes=100, **kwargs):
    """
    ResNeXt-20 (32x2d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=32, bottleneck_width=2,
                             model_name="resnext20_32x2d_cifar100", **kwargs)


def resnext20_32x2d_svhn(num_classes=10, **kwargs):
    """
    ResNeXt-20 (32x2d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=32, bottleneck_width=2,
                             model_name="resnext20_32x2d_svhn", **kwargs)


def resnext20_32x4d_cifar10(num_classes=10, **kwargs):
    """
    ResNeXt-20 (32x4d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=32, bottleneck_width=4,
                             model_name="resnext20_32x4d_cifar10", **kwargs)


def resnext20_32x4d_cifar100(num_classes=100, **kwargs):
    """
    ResNeXt-20 (32x4d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=32, bottleneck_width=4,
                             model_name="resnext20_32x4d_cifar100", **kwargs)


def resnext20_32x4d_svhn(num_classes=10, **kwargs):
    """
    ResNeXt-20 (32x4d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=20, cardinality=32, bottleneck_width=4,
                             model_name="resnext20_32x4d_svhn", **kwargs)


def resnext29_32x4d_cifar10(num_classes=10, **kwargs):
    """
    ResNeXt-29 (32x4d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=29, cardinality=32, bottleneck_width=4,
                             model_name="resnext29_32x4d_cifar10", **kwargs)


def resnext29_32x4d_cifar100(num_classes=100, **kwargs):
    """
    ResNeXt-29 (32x4d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=29, cardinality=32, bottleneck_width=4,
                             model_name="resnext29_32x4d_cifar100", **kwargs)


def resnext29_32x4d_svhn(num_classes=10, **kwargs):
    """
    ResNeXt-29 (32x4d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=29, cardinality=32, bottleneck_width=4,
                             model_name="resnext29_32x4d_svhn", **kwargs)


def resnext29_16x64d_cifar10(num_classes=10, **kwargs):
    """
    ResNeXt-29 (16x64d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=29, cardinality=16, bottleneck_width=64,
                             model_name="resnext29_16x64d_cifar10", **kwargs)


def resnext29_16x64d_cifar100(num_classes=100, **kwargs):
    """
    ResNeXt-29 (16x64d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=29, cardinality=16, bottleneck_width=64,
                             model_name="resnext29_16x64d_cifar100", **kwargs)


def resnext29_16x64d_svhn(num_classes=10, **kwargs):
    """
    ResNeXt-29 (16x64d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=29, cardinality=16, bottleneck_width=64,
                             model_name="resnext29_16x64d_svhn", **kwargs)


def resnext272_1x64d_cifar10(num_classes=10, **kwargs):
    """
    ResNeXt-272 (1x64d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=272, cardinality=1, bottleneck_width=64,
                             model_name="resnext272_1x64d_cifar10", **kwargs)


def resnext272_1x64d_cifar100(num_classes=100, **kwargs):
    """
    ResNeXt-272 (1x64d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=272, cardinality=1, bottleneck_width=64,
                             model_name="resnext272_1x64d_cifar100", **kwargs)


def resnext272_1x64d_svhn(num_classes=10, **kwargs):
    """
    ResNeXt-272 (1x64d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=272, cardinality=1, bottleneck_width=64,
                             model_name="resnext272_1x64d_svhn", **kwargs)


def resnext272_2x32d_cifar10(num_classes=10, **kwargs):
    """
    ResNeXt-272 (2x32d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=272, cardinality=2, bottleneck_width=32,
                             model_name="resnext272_2x32d_cifar10", **kwargs)


def resnext272_2x32d_cifar100(num_classes=100, **kwargs):
    """
    ResNeXt-272 (2x32d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=272, cardinality=2, bottleneck_width=32,
                             model_name="resnext272_2x32d_cifar100", **kwargs)


def resnext272_2x32d_svhn(num_classes=10, **kwargs):
    """
    ResNeXt-272 (2x32d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext_cifar(num_classes=num_classes, blocks=272, cardinality=2, bottleneck_width=32,
                             model_name="resnext272_2x32d_svhn", **kwargs)


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
        (resnext20_16x4d_cifar10, 10),
        (resnext20_16x4d_cifar100, 100),
        (resnext20_16x4d_svhn, 10),
        (resnext20_32x2d_cifar10, 10),
        (resnext20_32x2d_cifar100, 100),
        (resnext20_32x2d_svhn, 10),
        (resnext20_32x4d_cifar10, 10),
        (resnext20_32x4d_cifar100, 100),
        (resnext20_32x4d_svhn, 10),
        (resnext29_32x4d_cifar10, 10),
        (resnext29_32x4d_cifar100, 100),
        (resnext29_32x4d_svhn, 10),
        (resnext29_16x64d_cifar10, 10),
        (resnext29_16x64d_cifar100, 100),
        (resnext29_16x64d_svhn, 10),
        (resnext272_1x64d_cifar10, 10),
        (resnext272_1x64d_cifar100, 100),
        (resnext272_1x64d_svhn, 10),
        (resnext272_2x32d_cifar10, 10),
        (resnext272_2x32d_cifar100, 100),
        (resnext272_2x32d_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnext20_16x4d_cifar10 or weight_count == 1995082)
        assert (model != resnext20_16x4d_cifar100 or weight_count == 2087332)
        assert (model != resnext20_16x4d_svhn or weight_count == 1995082)
        assert (model != resnext20_32x2d_cifar10 or weight_count == 1946698)
        assert (model != resnext20_32x2d_cifar100 or weight_count == 2038948)
        assert (model != resnext20_32x2d_svhn or weight_count == 1946698)
        assert (model != resnext20_32x4d_cifar10 or weight_count == 3295562)
        assert (model != resnext20_32x4d_cifar100 or weight_count == 3387812)
        assert (model != resnext20_32x4d_svhn or weight_count == 3295562)
        assert (model != resnext29_32x4d_cifar10 or weight_count == 4775754)
        assert (model != resnext29_32x4d_cifar100 or weight_count == 4868004)
        assert (model != resnext29_32x4d_svhn or weight_count == 4775754)
        assert (model != resnext29_16x64d_cifar10 or weight_count == 68155210)
        assert (model != resnext29_16x64d_cifar100 or weight_count == 68247460)
        assert (model != resnext29_16x64d_svhn or weight_count == 68155210)
        assert (model != resnext272_1x64d_cifar10 or weight_count == 44540746)
        assert (model != resnext272_1x64d_cifar100 or weight_count == 44632996)
        assert (model != resnext272_1x64d_svhn or weight_count == 44540746)
        assert (model != resnext272_2x32d_cifar10 or weight_count == 32928586)
        assert (model != resnext272_2x32d_cifar100 or weight_count == 33020836)
        assert (model != resnext272_2x32d_svhn or weight_count == 32928586)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
