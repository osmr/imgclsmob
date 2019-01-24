"""
    DenseNet for CIFAR-10, implemented in Gluon.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

__all__ = ['CIFAR10DenseNet', 'densenet100_cifar10']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3
from .preresnet import PreResActivation
from .densenet import DenseUnit, TransitionBlock


class CIFAR10DenseNet(HybridBlock):
    """
    DenseNet model for CIFAR-10 from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bn_use_global_stats=False,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFAR10DenseNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(conv3x3(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    if i != 0:
                        stage.add(TransitionBlock(
                            in_channels=in_channels,
                            out_channels=(in_channels // 2),
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = in_channels // 2
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(DenseUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bn_use_global_stats=bn_use_global_stats,
                            dropout_rate=dropout_rate))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PreResActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_densenet_cifar10(blocks,
                         growth_rate,
                         model_name=None,
                         pretrained=False,
                         ctx=cpu(),
                         root=os.path.join('~', '.mxnet', 'models'),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def densenet100_cifar10(**kwargs):
    """
    DenseNet-100 model for CIFAR-10 from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_densenet_cifar10(blocks=100, growth_rate=12, model_name="densenet100_cifar10", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        densenet100_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != densenet100_cifar10 or weight_count == 769162)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 10))


if __name__ == "__main__":
    _test()
