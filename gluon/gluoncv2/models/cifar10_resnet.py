"""
    CIFAR-10 ResNet, implemented in Gluon.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['CIFAR10ResNet', 'cifar10_resnet20', 'cifar10_resnet56', 'cifar10_resnet110']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3_block
from .resnet import ResUnit


class CIFAR10ResNet(HybridBlock):
    """
    CIFAR-10 ResNet model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 100
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFAR10ResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None,
                activate=False))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(ResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=False,
                            conv1_stride=False))
                        in_channels = out_channels
                self.features.add(stage)
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


def get_cifar_resnet(blocks,
                     model_name=None,
                     pretrained=False,
                     ctx=cpu(),
                     root=os.path.join('~', '.mxnet', 'models'),
                     **kwargs):
    """
    Create CIFAR-10 ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    assert ((blocks - 2) % 6 == 0)
    init_block_channels = 16
    channels_per_layers = [16, 32, 64]
    layers = [(blocks - 2) // 6] * 3

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFAR10ResNet(
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


def cifar10_resnet20(**kwargs):
    """
    CIFAR-10 ResNet-20 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(blocks=20, model_name="cifar10_resnet20", **kwargs)


def cifar10_resnet56(**kwargs):
    """
    CIFAR-10 ResNet-56 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(blocks=56, model_name="cifar10_resnet56", **kwargs)


def cifar10_resnet110(**kwargs):
    """
    CIFAR-10 ResNet-110 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(blocks=110, model_name="cifar10_resnet110", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        cifar10_resnet20,
        cifar10_resnet56,
        cifar10_resnet110,
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
        assert (model != cifar10_resnet20 or weight_count == 272474)
        assert (model != cifar10_resnet56 or weight_count == 855770)
        assert (model != cifar10_resnet110 or weight_count == 1730714)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 10))


if __name__ == "__main__":
    _test()
