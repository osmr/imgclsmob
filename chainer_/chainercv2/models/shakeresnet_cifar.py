"""
    ShakeResNet for CIFAR, implemented in Chainer.
    Original paper: 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.
"""

__all__ = ['CIFARShakeResNet', 'shakeresnet20_cifar10', 'shakeresnet20_cifar100']

import os
import chainer
from chainer import cuda
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, SimpleSequential
from .resnet import ResBlock, ResBottleneck


class ShakeShake(chainer.function.Function):
    """
    Shake-Shake function.
    """

    def forward(self, inputs):
        x1, x2 = inputs
        if configuration.config.train:
            xp = cuda.get_array_module(x1)
            alpha = xp.empty((x1.shape[0], 1, 1, 1), dtype=x1.dtype)
            for i in range(len(alpha)):
                alpha[i] = xp.random.rand()
            return alpha * x1 + (1 - alpha) * x2,
        else:
            return 0.5 * (x1 + x2),

    def backward(self, inputs, grad_outputs):
        dy, = grad_outputs
        xp = cuda.get_array_module(dy)
        beta = xp.empty((dy.shape[0], 1, 1, 1), dtype=dy.dtype)
        for i in range(len(beta)):
            beta[i] = xp.random.rand()
        return beta * dy, (xp.ones(dy.shape, dtype=dy.dtype) - beta) * dy


class ShakeResUnit(Chain):
    """
    ShakeResNet unit with residual connection.

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
        super(ShakeResUnit, self).__init__()
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
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=None,
                    activate=False)
            self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = ShakeShake()(x1, x2) + identity
        x = self.activ(x)
        return x


class CIFARShakeResNet(Chain):
    """
    ResNet model for CIFAR from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

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
        super(CIFARShakeResNet, self).__init__()
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
                            setattr(stage, "unit{}".format(j + 1), ShakeResUnit(
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


def get_shakeresnet_cifar(classes,
                          blocks,
                          bottleneck,
                          model_name=None,
                          pretrained=False,
                          root=os.path.join('~', '.chainer', 'models'),
                          **kwargs):
    """
    Create ShakeResNet model for CIFAR with specific parameters.

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

    net = CIFARShakeResNet(
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


def shakeresnet20_cifar10(classes=10, **kwargs):
    """
    ShakeResNet-20 model for CIFAR-10 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="shakeresnet20_cifar10",
                                 **kwargs)


def shakeresnet20_cifar100(classes=100, **kwargs):
    """
    ShakeResNet-20 model for CIFAR-100 from 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakeresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="shakeresnet20_cifar100",
                                 **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = True

    pretrained = False

    models = [
        (shakeresnet20_cifar10, 10),
        (shakeresnet20_cifar100, 100),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shakeresnet20_cifar10 or weight_count == 541082)
        assert (model != shakeresnet20_cifar100 or weight_count == 546932)

        x = np.zeros((14, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (14, classes))


if __name__ == "__main__":
    _test()
