"""
    WRN for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['CIFARWRN', 'wrn16_10_cifar10', 'wrn16_10_cifar100', 'wrn16_10_svhn', 'wrn28_10_cifar10',
           'wrn28_10_cifar100', 'wrn28_10_svhn', 'wrn40_8_cifar10', 'wrn40_8_cifar100', 'wrn40_8_svhn']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, SimpleSequential
from .preresnet import PreResUnit, PreResActivation


class CIFARWRN(Chain):
    """
    WRN model for CIFAR from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
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
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARWRN, self).__init__()
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
                                bottleneck=False,
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


def get_wrn_cifar(classes,
                  blocks,
                  width_factor,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create WRN model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    width_factor : int
        Wide scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    assert ((blocks - 4) % 6 == 0)
    layers = [(blocks - 4) // 6] * 3
    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci * width_factor] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFARWRN(
        channels=channels,
        init_block_channels=init_block_channels,
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


def wrn16_10_cifar10(classes=10, **kwargs):
    """
    WRN-16-10 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=16, width_factor=10, model_name="wrn16_10_cifar10", **kwargs)


def wrn16_10_cifar100(classes=100, **kwargs):
    """
    WRN-16-10 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=16, width_factor=10, model_name="wrn16_10_cifar100", **kwargs)


def wrn16_10_svhn(classes=10, **kwargs):
    """
    WRN-16-10 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=16, width_factor=10, model_name="wrn16_10_svhn", **kwargs)


def wrn28_10_cifar10(classes=10, **kwargs):
    """
    WRN-28-10 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=28, width_factor=10, model_name="wrn28_10_cifar10", **kwargs)


def wrn28_10_cifar100(classes=100, **kwargs):
    """
    WRN-28-10 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=28, width_factor=10, model_name="wrn28_10_cifar100", **kwargs)


def wrn28_10_svhn(classes=10, **kwargs):
    """
    WRN-28-10 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=28, width_factor=10, model_name="wrn28_10_svhn", **kwargs)


def wrn40_8_cifar10(classes=10, **kwargs):
    """
    WRN-40-8 model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=40, width_factor=8, model_name="wrn40_8_cifar10", **kwargs)


def wrn40_8_cifar100(classes=100, **kwargs):
    """
    WRN-40-8 model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=40, width_factor=8, model_name="wrn40_8_cifar100", **kwargs)


def wrn40_8_svhn(classes=10, **kwargs):
    """
    WRN-40-8 model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=40, width_factor=8, model_name="wrn40_8_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (wrn16_10_cifar10, 10),
        (wrn16_10_cifar100, 100),
        (wrn16_10_svhn, 10),
        (wrn28_10_cifar10, 10),
        (wrn28_10_cifar100, 100),
        (wrn28_10_svhn, 10),
        (wrn40_8_cifar10, 10),
        (wrn40_8_cifar100, 100),
        (wrn40_8_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn16_10_cifar10 or weight_count == 17116634)
        assert (model != wrn16_10_cifar100 or weight_count == 17174324)
        assert (model != wrn16_10_svhn or weight_count == 17116634)
        assert (model != wrn28_10_cifar10 or weight_count == 36479194)
        assert (model != wrn28_10_cifar100 or weight_count == 36536884)
        assert (model != wrn28_10_svhn or weight_count == 36479194)
        assert (model != wrn40_8_cifar10 or weight_count == 35748314)
        assert (model != wrn40_8_cifar100 or weight_count == 35794484)
        assert (model != wrn40_8_svhn or weight_count == 35748314)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
