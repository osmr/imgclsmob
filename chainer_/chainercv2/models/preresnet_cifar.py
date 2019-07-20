"""
    PreResNet for CIFAR/SVHN, implemented in Chainer.
    Original papers: 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
"""

__all__ = ['CIFARPreResNet', 'preresnet20_cifar10', 'preresnet20_cifar100', 'preresnet20_svhn',
           'preresnet56_cifar10', 'preresnet56_cifar100', 'preresnet56_svhn',
           'preresnet110_cifar10', 'preresnet110_cifar100', 'preresnet110_svhn',
           'preresnet164bn_cifar10', 'preresnet164bn_cifar100', 'preresnet164bn_svhn',
           'preresnet272bn_cifar10', 'preresnet272bn_cifar100', 'preresnet272bn_svhn',
           'preresnet542bn_cifar10', 'preresnet542bn_cifar100', 'preresnet542bn_svhn',
           'preresnet1001_cifar10', 'preresnet1001_cifar100', 'preresnet1001_svhn',
           'preresnet1202_cifar10', 'preresnet1202_cifar100', 'preresnet1202_svhn']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, SimpleSequential
from .preresnet import PreResUnit, PreResActivation


class CIFARPreResNet(Chain):
    """
    PreResNet model for CIFAR from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
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
        super(CIFARPreResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), PreResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=False))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(
                    in_channels=in_channels))
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


def get_preresnet_cifar(classes,
                        blocks,
                        bottleneck,
                        model_name=None,
                        pretrained=False,
                        root=os.path.join("~", ".chainer", "models"),
                        **kwargs):
    """
    Create PreResNet model for CIFAR with specific parameters.

    Parameters:
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

    net = CIFARPreResNet(
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


def preresnet20_cifar10(classes=10, **kwargs):
    """
    PreResNet-20 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="preresnet20_cifar10", **kwargs)


def preresnet20_cifar100(classes=100, **kwargs):
    """
    PreResNet-20 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="preresnet20_cifar100",
                               **kwargs)


def preresnet20_svhn(classes=10, **kwargs):
    """
    PreResNet-20 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="preresnet20_svhn", **kwargs)


def preresnet56_cifar10(classes=10, **kwargs):
    """
    PreResNet-56 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="preresnet56_cifar10", **kwargs)


def preresnet56_cifar100(classes=100, **kwargs):
    """
    PreResNet-56 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="preresnet56_cifar100",
                               **kwargs)


def preresnet56_svhn(classes=10, **kwargs):
    """
    PreResNet-56 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=56, bottleneck=False, model_name="preresnet56_svhn", **kwargs)


def preresnet110_cifar10(classes=10, **kwargs):
    """
    PreResNet-110 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="preresnet110_cifar10",
                               **kwargs)


def preresnet110_cifar100(classes=100, **kwargs):
    """
    PreResNet-110 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="preresnet110_cifar100",
                               **kwargs)


def preresnet110_svhn(classes=10, **kwargs):
    """
    PreResNet-110 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=110, bottleneck=False, model_name="preresnet110_svhn",
                               **kwargs)


def preresnet164bn_cifar10(classes=10, **kwargs):
    """
    PreResNet-164(BN) model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="preresnet164bn_cifar10",
                               **kwargs)


def preresnet164bn_cifar100(classes=100, **kwargs):
    """
    PreResNet-164(BN) model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="preresnet164bn_cifar100",
                               **kwargs)


def preresnet164bn_svhn(classes=10, **kwargs):
    """
    PreResNet-164(BN) model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=164, bottleneck=True, model_name="preresnet164bn_svhn",
                               **kwargs)


def preresnet272bn_cifar10(classes=10, **kwargs):
    """
    PreResNet-272(BN) model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="preresnet272bn_cifar10",
                               **kwargs)


def preresnet272bn_cifar100(classes=100, **kwargs):
    """
    PreResNet-272(BN) model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="preresnet272bn_cifar100",
                               **kwargs)


def preresnet272bn_svhn(classes=10, **kwargs):
    """
    PreResNet-272(BN) model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=272, bottleneck=True, model_name="preresnet272bn_svhn",
                               **kwargs)


def preresnet542bn_cifar10(classes=10, **kwargs):
    """
    PreResNet-542(BN) model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="preresnet542bn_cifar10",
                               **kwargs)


def preresnet542bn_cifar100(classes=100, **kwargs):
    """
    PreResNet-542(BN) model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="preresnet542bn_cifar100",
                               **kwargs)


def preresnet542bn_svhn(classes=10, **kwargs):
    """
    PreResNet-542(BN) model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=542, bottleneck=True, model_name="preresnet542bn_svhn",
                               **kwargs)


def preresnet1001_cifar10(classes=10, **kwargs):
    """
    PreResNet-1001 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="preresnet1001_cifar10",
                               **kwargs)


def preresnet1001_cifar100(classes=100, **kwargs):
    """
    PreResNet-1001 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="preresnet1001_cifar100",
                               **kwargs)


def preresnet1001_svhn(classes=10, **kwargs):
    """
    PreResNet-1001 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1001, bottleneck=True, model_name="preresnet1001_svhn",
                               **kwargs)


def preresnet1202_cifar10(classes=10, **kwargs):
    """
    PreResNet-1202 model for CIFAR-10 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="preresnet1202_cifar10",
                               **kwargs)


def preresnet1202_cifar100(classes=100, **kwargs):
    """
    PreResNet-1202 model for CIFAR-100 from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="preresnet1202_cifar100",
                               **kwargs)


def preresnet1202_svhn(classes=10, **kwargs):
    """
    PreResNet-1202 model for SVHN from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet_cifar(classes=classes, blocks=1202, bottleneck=False, model_name="preresnet1202_svhn",
                               **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (preresnet20_cifar10, 10),
        (preresnet20_cifar100, 100),
        (preresnet20_svhn, 10),
        (preresnet56_cifar10, 10),
        (preresnet56_cifar100, 100),
        (preresnet56_svhn, 10),
        (preresnet110_cifar10, 10),
        (preresnet110_cifar100, 100),
        (preresnet110_svhn, 10),
        (preresnet164bn_cifar10, 10),
        (preresnet164bn_cifar100, 100),
        (preresnet164bn_svhn, 10),
        (preresnet272bn_cifar10, 10),
        (preresnet272bn_cifar100, 100),
        (preresnet272bn_svhn, 10),
        (preresnet542bn_cifar10, 10),
        (preresnet542bn_cifar100, 100),
        (preresnet542bn_svhn, 10),
        (preresnet1001_cifar10, 10),
        (preresnet1001_cifar100, 100),
        (preresnet1001_svhn, 10),
        (preresnet1202_cifar10, 10),
        (preresnet1202_cifar100, 100),
        (preresnet1202_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != preresnet20_cifar10 or weight_count == 272282)
        assert (model != preresnet20_cifar100 or weight_count == 278132)
        assert (model != preresnet20_svhn or weight_count == 272282)
        assert (model != preresnet56_cifar10 or weight_count == 855578)
        assert (model != preresnet56_cifar100 or weight_count == 861428)
        assert (model != preresnet56_svhn or weight_count == 855578)
        assert (model != preresnet110_cifar10 or weight_count == 1730522)
        assert (model != preresnet110_cifar100 or weight_count == 1736372)
        assert (model != preresnet110_svhn or weight_count == 1730522)
        assert (model != preresnet164bn_cifar10 or weight_count == 1703258)
        assert (model != preresnet164bn_cifar100 or weight_count == 1726388)
        assert (model != preresnet164bn_svhn or weight_count == 1703258)
        assert (model != preresnet272bn_cifar10 or weight_count == 2816090)
        assert (model != preresnet272bn_cifar100 or weight_count == 2839220)
        assert (model != preresnet272bn_svhn or weight_count == 2816090)
        assert (model != preresnet542bn_cifar10 or weight_count == 5598170)
        assert (model != preresnet542bn_cifar100 or weight_count == 5621300)
        assert (model != preresnet542bn_svhn or weight_count == 5598170)
        assert (model != preresnet1001_cifar10 or weight_count == 10327706)
        assert (model != preresnet1001_cifar100 or weight_count == 10350836)
        assert (model != preresnet1001_svhn or weight_count == 10327706)
        assert (model != preresnet1202_cifar10 or weight_count == 19423834)
        assert (model != preresnet1202_cifar100 or weight_count == 19429684)
        assert (model != preresnet1202_svhn or weight_count == 19423834)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
