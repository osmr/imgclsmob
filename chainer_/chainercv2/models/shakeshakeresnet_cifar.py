"""
    Shake-Shake-ResNet for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.
"""

__all__ = ['CIFARShakeShakeResNet', 'shakeshakeresnet20_2x16d_cifar10', 'shakeshakeresnet20_2x16d_cifar100',
           'shakeshakeresnet20_2x16d_svhn', 'shakeshakeresnet26_2x32d_cifar10', 'shakeshakeresnet26_2x32d_cifar100',
           'shakeshakeresnet26_2x32d_svhn']

import os
import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv3x3_block, SimpleSequential
from .resnet import ResBlock, ResBottleneck


class ShakeShake(chainer.function.Function):
    """
    Shake-Shake function.
    """

    def forward(self, inputs):
        x1, x2 = inputs
        if chainer.config.train:
            xp = backend.get_array_module(x1)
            alpha = xp.empty((x1.shape[0], 1, 1, 1), dtype=x1.dtype)
            for i in range(len(alpha)):
                alpha[i] = xp.random.rand()
            return alpha * x1 + (1 - alpha) * x2,
        else:
            return 0.5 * (x1 + x2),

    def backward(self, inputs, grad_outputs):
        dy, = grad_outputs
        xp = backend.get_array_module(dy)
        beta = xp.empty((dy.shape[0], 1, 1, 1), dtype=dy.dtype)
        for i in range(len(beta)):
            beta[i] = xp.random.rand()
        return beta * dy, (xp.ones(dy.shape, dtype=dy.dtype) - beta) * dy


class ShakeShakeShortcut(Chain):
    """
    Shake-Shake-ResNet shortcut.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(ShakeShakeShortcut, self).__init__()
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        with self.init_scope():
            self.pool = partial(
                F.average_pooling_2d,
                ksize=1,
                stride=stride)
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)

    def __call__(self, x):
        x1 = self.pool(x)
        x1 = self.conv1(x1)
        x2 = x[:, :, :-1, :-1]
        x2 = F.pad(x2, pad_width=((0, 0), (0, 0), (1, 0), (1, 0)), mode="constant", constant_values=0)
        x2 = self.pool(x2)
        x2 = self.conv2(x2)
        x = F.concat((x1, x2), axis=1)
        x = self.bn(x)
        return x


class ShakeShakeResUnit(Chain):
    """
    Shake-Shake-ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(ShakeShakeResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        branch_class = ResBottleneck if bottleneck else ResBlock

        with self.init_scope():
            self.branch1 = branch_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            self.branch2 = branch_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            if self.resize_identity:
                self.identity_branch = ShakeShakeShortcut(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)
            self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_branch(x)
        else:
            identity = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = ShakeShake()(x1, x2) + identity
        x = self.activ(x)
        return x


class CIFARShakeShakeResNet(Chain):
    """
    Shake-Shake-ResNet model for CIFAR from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

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
        super(CIFARShakeShakeResNet, self).__init__()
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
                            setattr(stage, "unit{}".format(j + 1), ShakeShakeResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck))
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


def get_shakeshakeresnet_cifar(classes,
                               blocks,
                               bottleneck,
                               first_stage_channels=16,
                               model_name=None,
                               pretrained=False,
                               root=os.path.join("~", ".chainer", "models"),
                               **kwargs):
    """
    Create Shake-Shake-ResNet model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    first_stage_channels : int, default 16
        Number of output channels for the first stage.
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

    init_block_channels = 16

    from functools import reduce
    channels_per_layers = reduce(lambda x, y: x + [x[-1] * 2], range(2), [first_stage_channels])

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = CIFARShakeShakeResNet(
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


def shakeshakeresnet20_2x16d_cifar10(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-20-2x16d model for CIFAR-10 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, first_stage_channels=16,
                                      model_name="shakeshakeresnet20_2x16d_cifar10", **kwargs)


def shakeshakeresnet20_2x16d_cifar100(classes=100, **kwargs):
    """
    Shake-Shake-ResNet-20-2x16d model for CIFAR-100 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, first_stage_channels=16,
                                      model_name="shakeshakeresnet20_2x16d_cifar100", **kwargs)


def shakeshakeresnet20_2x16d_svhn(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-20-2x16d model for SVHN from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, first_stage_channels=16,
                                      model_name="shakeshakeresnet20_2x16d_svhn", **kwargs)


def shakeshakeresnet26_2x32d_cifar10(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-26-2x32d model for CIFAR-10 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=26, bottleneck=False, first_stage_channels=32,
                                      model_name="shakeshakeresnet26_2x32d_cifar10", **kwargs)


def shakeshakeresnet26_2x32d_cifar100(classes=100, **kwargs):
    """
    Shake-Shake-ResNet-26-2x32d model for CIFAR-100 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=26, bottleneck=False, first_stage_channels=32,
                                      model_name="shakeshakeresnet26_2x32d_cifar100", **kwargs)


def shakeshakeresnet26_2x32d_svhn(classes=10, **kwargs):
    """
    Shake-Shake-ResNet-26-2x32d model for SVHN from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=26, bottleneck=False, first_stage_channels=32,
                                      model_name="shakeshakeresnet26_2x32d_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (shakeshakeresnet20_2x16d_cifar10, 10),
        (shakeshakeresnet20_2x16d_cifar100, 100),
        (shakeshakeresnet20_2x16d_svhn, 10),
        (shakeshakeresnet26_2x32d_cifar10, 10),
        (shakeshakeresnet26_2x32d_cifar100, 100),
        (shakeshakeresnet26_2x32d_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shakeshakeresnet20_2x16d_cifar10 or weight_count == 541082)
        assert (model != shakeshakeresnet20_2x16d_cifar100 or weight_count == 546932)
        assert (model != shakeshakeresnet20_2x16d_svhn or weight_count == 541082)
        assert (model != shakeshakeresnet26_2x32d_cifar10 or weight_count == 2923162)
        assert (model != shakeshakeresnet26_2x32d_cifar100 or weight_count == 2934772)
        assert (model != shakeshakeresnet26_2x32d_svhn or weight_count == 2923162)

        x = np.zeros((14, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (14, classes))


if __name__ == "__main__":
    _test()
