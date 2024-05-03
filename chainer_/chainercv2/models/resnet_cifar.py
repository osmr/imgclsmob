"""
    ResNet for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['CIFARResNet', 'resnet20_cifar10', 'resnet20_cifar100', 'resnet20_svhn',
           'resnet56_cifar10', 'resnet56_cifar100', 'resnet56_svhn',
           'resnet110_cifar10', 'resnet110_cifar100', 'resnet110_svhn',
           'resnet164bn_cifar10', 'resnet164bn_cifar100', 'resnet164bn_svhn',
           'resnet272bn_cifar10', 'resnet272bn_cifar100', 'resnet272bn_svhn',
           'resnet542bn_cifar10', 'resnet542bn_cifar100', 'resnet542bn_svhn',
           'resnet1001_cifar10', 'resnet1001_cifar100', 'resnet1001_svhn',
           'resnet1202_cifar10', 'resnet1202_cifar100', 'resnet1202_svhn']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3_block, SimpleSequential
from .resnet import ResUnit


class CIFARResNet(Chain):
    """
    ResNet model for CIFAR from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), ResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=False))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_resnet_cifar(classes,
                     blocks,
                     bottleneck,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".chainer", "models"),
                     **kwargs):
    """
    Create ResNet model for CIFAR with specific parameters.

    Parameters
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    assert (classes in [10, 100])

    if bottleneck:
        assert ((blocks - 2) % 9 == 0)
        layers = [(blocks - 2) // 9] * 3
    else:
        assert ((blocks - 2) % 6 == 0)
        layers = [(blocks - 2) // 6] * 3

    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = CIFARResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        classes=classes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def resnet20_cifar10(classes=10, **kwargs):
    """
    ResNet-20 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resnet20_cifar10", **kwargs)


def resnet20_cifar100(classes=100, **kwargs):
    """
    ResNet-20 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resnet20_cifar100", **kwargs)


def resnet20_svhn(classes=10, **kwargs):
    """
    ResNet-20 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resnet20_svhn", **kwargs)


def resnet56_cifar10(classes=10, **kwargs):
    """
    ResNet-56 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="resnet56_cifar10", **kwargs)


def resnet56_cifar100(classes=100, **kwargs):
    """
    ResNet-56 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="resnet56_cifar100", **kwargs)


def resnet56_svhn(classes=10, **kwargs):
    """
    ResNet-56 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="resnet56_svhn", **kwargs)


def resnet110_cifar10(classes=10, **kwargs):
    """
    ResNet-110 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="resnet110_cifar10", **kwargs)


def resnet110_cifar100(classes=100, **kwargs):
    """
    ResNet-110 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="resnet110_cifar100", **kwargs)


def resnet110_svhn(classes=10, **kwargs):
    """
    ResNet-110 model for SVHN from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="resnet110_svhn", **kwargs)


def resnet164bn_cifar10(classes=10, **kwargs):
    """
    ResNet-164(BN) model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="resnet164bn_cifar10", **kwargs)


def resnet164bn_cifar100(classes=100, **kwargs):
    """
    ResNet-164(BN) model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="resnet164bn_cifar100", **kwargs)


def resnet164bn_svhn(classes=10, **kwargs):
    """
    ResNet-164(BN) model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="resnet164bn_svhn", **kwargs)


def resnet272bn_cifar10(classes=10, **kwargs):
    """
    ResNet-272(BN) model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="resnet272bn_cifar10", **kwargs)


def resnet272bn_cifar100(classes=100, **kwargs):
    """
    ResNet-272(BN) model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="resnet272bn_cifar100", **kwargs)


def resnet272bn_svhn(classes=10, **kwargs):
    """
    ResNet-272(BN) model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="resnet272bn_svhn", **kwargs)


def resnet542bn_cifar10(classes=10, **kwargs):
    """
    ResNet-542(BN) model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="resnet542bn_cifar10", **kwargs)


def resnet542bn_cifar100(classes=100, **kwargs):
    """
    ResNet-542(BN) model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="resnet542bn_cifar100", **kwargs)


def resnet542bn_svhn(classes=10, **kwargs):
    """
    ResNet-542(BN) model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="resnet542bn_svhn", **kwargs)


def resnet1001_cifar10(classes=10, **kwargs):
    """
    ResNet-1001 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="resnet1001_cifar10", **kwargs)


def resnet1001_cifar100(classes=100, **kwargs):
    """
    ResNet-1001 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="resnet1001_cifar100", **kwargs)


def resnet1001_svhn(classes=10, **kwargs):
    """
    ResNet-1001 model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="resnet1001_svhn", **kwargs)


def resnet1202_cifar10(classes=10, **kwargs):
    """
    ResNet-1202 model for CIFAR-10 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="resnet1202_cifar10", **kwargs)


def resnet1202_cifar100(classes=100, **kwargs):
    """
    ResNet-1202 model for CIFAR-100 from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="resnet1202_cifar100", **kwargs)


def resnet1202_svhn(classes=10, **kwargs):
    """
    ResNet-1202 model for SVHN from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="resnet1202_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (resnet20_cifar10, 10),
        (resnet20_cifar100, 100),
        (resnet20_svhn, 10),
        (resnet56_cifar10, 10),
        (resnet56_cifar100, 100),
        (resnet56_svhn, 10),
        (resnet110_cifar10, 10),
        (resnet110_cifar100, 100),
        (resnet110_svhn, 10),
        (resnet164bn_cifar10, 10),
        (resnet164bn_cifar100, 100),
        (resnet164bn_svhn, 10),
        (resnet272bn_cifar10, 10),
        (resnet272bn_cifar100, 100),
        (resnet272bn_svhn, 10),
        (resnet542bn_cifar10, 10),
        (resnet542bn_cifar100, 100),
        (resnet542bn_svhn, 10),
        (resnet1001_cifar10, 10),
        (resnet1001_cifar100, 100),
        (resnet1001_svhn, 10),
        (resnet1202_cifar10, 10),
        (resnet1202_cifar100, 100),
        (resnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnet20_cifar10 or weight_count == 272474)
        assert (model != resnet20_cifar100 or weight_count == 278324)
        assert (model != resnet20_svhn or weight_count == 272474)
        assert (model != resnet56_cifar10 or weight_count == 855770)
        assert (model != resnet56_cifar100 or weight_count == 861620)
        assert (model != resnet56_svhn or weight_count == 855770)
        assert (model != resnet110_cifar10 or weight_count == 1730714)
        assert (model != resnet110_cifar100 or weight_count == 1736564)
        assert (model != resnet110_svhn or weight_count == 1730714)
        assert (model != resnet164bn_cifar10 or weight_count == 1704154)
        assert (model != resnet164bn_cifar100 or weight_count == 1727284)
        assert (model != resnet164bn_svhn or weight_count == 1704154)
        assert (model != resnet272bn_cifar10 or weight_count == 2816986)
        assert (model != resnet272bn_cifar100 or weight_count == 2840116)
        assert (model != resnet272bn_svhn or weight_count == 2816986)
        assert (model != resnet542bn_cifar10 or weight_count == 5599066)
        assert (model != resnet542bn_cifar100 or weight_count == 5622196)
        assert (model != resnet542bn_svhn or weight_count == 5599066)
        assert (model != resnet1001_cifar10 or weight_count == 10328602)
        assert (model != resnet1001_cifar100 or weight_count == 10351732)
        assert (model != resnet1001_svhn or weight_count == 10328602)
        assert (model != resnet1202_cifar10 or weight_count == 19424026)
        assert (model != resnet1202_cifar100 or weight_count == 19429876)
        assert (model != resnet1202_svhn or weight_count == 19424026)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
