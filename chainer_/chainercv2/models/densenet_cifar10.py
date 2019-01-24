"""
    DenseNet for CIFAR-10, implemented in Chainer.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

__all__ = ['CIFAR10DenseNet', 'densenet100_cifar10']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, SimpleSequential
from .preresnet import PreResActivation
from .densenet import DenseUnit, TransitionBlock


class CIFAR10DenseNet(Chain):
    """
    DenseNet model for CIFAR-10 from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFAR10DenseNet, self).__init__()
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
                        if i != 0:
                            setattr(stage, "trans{}".format(i + 1), TransitionBlock(
                                in_channels=in_channels,
                                out_channels=(in_channels // 2)))
                            in_channels = in_channels // 2
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), DenseUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                dropout_rate=dropout_rate))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(in_channels=in_channels))
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


def get_densenet_cifar10(blocks,
                         growth_rate,
                         model_name=None,
                         pretrained=False,
                         root=os.path.join('~', '.chainer', 'models'),
                         **kwargs):
    """
    Create DenseNet model for CIFAR-10 with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    growth_rate : int
        Growth rate.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    assert ((blocks - 4) % 6 == 0)
    layers = [(blocks - 4) // 6] * 3
    init_block_channels = 2 * growth_rate

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = CIFAR10DenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def densenet100_cifar10(**kwargs):
    """
    DenseNet-100 model for CIFAR-10 from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar10(blocks=100, growth_rate=12, model_name="densenet100_cifar10", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        densenet100_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != densenet100_cifar10 or weight_count == 769162)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, 10))


if __name__ == "__main__":
    _test()
