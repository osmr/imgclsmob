"""
    ShakeDrop-ResNet for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'ShakeDrop Regularization for Deep Residual Learning,' https://arxiv.org/abs/1802.02375.
"""

__all__ = ['CIFARShakeDropResNet', 'shakedropresnet20_cifar10', 'shakedropresnet20_cifar100', 'shakedropresnet20_svhn']

import os
import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, SimpleSequential
from .resnet import ResBlock, ResBottleneck


class ShakeDrop(chainer.function.Function):
    """
    ShakeDrop function.

    Parameters:
    ----------
    p : float
        ShakeDrop specific probability (of life) for Bernoulli random variable.
    """
    def __init__(self, p):
        super(ShakeDrop, self).__init__()
        self.p = p
        self.b = None

    def forward(self, inputs):
        x, = inputs
        if chainer.config.train:
            xp = backend.get_array_module(x)
            b = xp.random.binomial(n=1, p=self.p)
            alpha = xp.empty((x.shape[0], 1, 1, 1), dtype=x.dtype)
            for i in range(len(alpha)):
                alpha[i] = xp.random.uniform(low=-1.0, high=1.0)
            y = (b + alpha - b * alpha) * x
            self.b = b
        else:
            y = self.p * x
        return y,

    def backward(self, inputs, grad_outputs):
        dy, = grad_outputs
        b = self.b
        xp = backend.get_array_module(dy)
        beta = xp.empty((dy.shape[0], 1, 1, 1), dtype=dy.dtype)
        for i in range(len(beta)):
            beta[i] = xp.random.uniform(low=0.0, high=1.0)
        return (b + beta - b * beta) * dy,


class ShakeDropResUnit(Chain):
    """
    ShakeDrop-ResNet unit with residual connection.

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
    life_prob : float
        Residual branch life probability.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 life_prob):
        super(ShakeDropResUnit, self).__init__()
        self.life_prob = life_prob
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        body_class = ResBottleneck if bottleneck else ResBlock

        with self.init_scope():
            self.body = body_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=None)
            self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = ShakeDrop(self.life_prob)(x) + identity
        # x = self.shake_drop(x) + identity
        x = self.activ(x)
        return x


class CIFARShakeDropResNet(Chain):
    """
    ShakeDrop-ResNet model for CIFAR from 'ShakeDrop Regularization for Deep Residual Learning,'
    https://arxiv.org/abs/1802.02375.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    life_probs : list of float
        Residual branch life probability for each unit.
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
                 life_probs,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARShakeDropResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                k = 0
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), ShakeDropResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                life_prob=life_probs[k]))
                            in_channels = out_channels
                            k += 1
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


def get_shakedropresnet_cifar(classes,
                              blocks,
                              bottleneck,
                              model_name=None,
                              pretrained=False,
                              root=os.path.join("~", ".chainer", "models"),
                              **kwargs):
    """
    Create ShakeDrop-ResNet model for CIFAR with specific parameters.

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

    init_block_channels = 16
    channels_per_layers = [16, 32, 64]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    total_layers = sum(layers)
    final_death_prob = 0.5
    life_probs = [1.0 - float(i + 1) / float(total_layers) * final_death_prob for i in range(total_layers)]

    net = CIFARShakeDropResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        life_probs=life_probs,
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


def shakedropresnet20_cifar10(classes=10, **kwargs):
    """
    ShakeDrop-ResNet-20 model for CIFAR-10 from 'ShakeDrop Regularization for Deep Residual Learning,'
    https://arxiv.org/abs/1802.02375.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakedropresnet_cifar(classes=classes, blocks=20, bottleneck=False,
                                     model_name="shakedropresnet20_cifar10", **kwargs)


def shakedropresnet20_cifar100(classes=100, **kwargs):
    """
    ShakeDrop-ResNet-20 model for CIFAR-100 from 'ShakeDrop Regularization for Deep Residual Learning,'
    https://arxiv.org/abs/1802.02375.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakedropresnet_cifar(classes=classes, blocks=20, bottleneck=False,
                                     model_name="shakedropresnet20_cifar100", **kwargs)


def shakedropresnet20_svhn(classes=10, **kwargs):
    """
    ShakeDrop-ResNet-20 model for SVHN from 'ShakeDrop Regularization for Deep Residual Learning,'
    https://arxiv.org/abs/1802.02375.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_shakedropresnet_cifar(classes=classes, blocks=20, bottleneck=False,
                                     model_name="shakedropresnet20_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (shakedropresnet20_cifar10, 10),
        (shakedropresnet20_cifar100, 100),
        (shakedropresnet20_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != shakedropresnet20_cifar10 or weight_count == 272474)
        assert (model != shakedropresnet20_cifar100 or weight_count == 278324)
        assert (model != shakedropresnet20_svhn or weight_count == 272474)

        x = np.zeros((14, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (14, classes))


if __name__ == "__main__":
    _test()
