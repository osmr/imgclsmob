"""
    Shake-Shake-ResNet for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Shake-Shake regularization,' https://arxiv.org/abs/1705.07485.
"""

__all__ = ['CIFARShakeShakeResNet', 'shakeshakeresnet20_2x16d_cifar10', 'shakeshakeresnet20_2x16d_cifar100',
           'shakeshakeresnet20_2x16d_svhn', 'shakeshakeresnet26_2x32d_cifar10', 'shakeshakeresnet26_2x32d_cifar100',
           'shakeshakeresnet26_2x32d_svhn']

import os
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv3x3_block
from .resnet import ResBlock, ResBottleneck


class ShakeShake(mx.autograd.Function):
    """
    Shake-Shake function.
    """

    def forward(self, x1, x2):
        if mx.autograd.is_training():
            alpha = mx.nd.random.uniform_like(x1.slice(begin=(None, 0, 0, 0), end=(None, 1, 1, 1)))
            y = mx.nd.broadcast_mul(alpha, x1) + mx.nd.broadcast_mul(1 - alpha, x2)
        else:
            y = 0.5 * (x1 + x2)
        return y

    def backward(self, dy):
        beta = mx.nd.random.uniform_like(dy.slice(begin=(None, 0, 0, 0), end=(None, 1, 1, 1)))
        return mx.nd.broadcast_mul(beta, dy), mx.nd.broadcast_mul(1 - beta, dy)


class ShakeShakeShortcut(HybridBlock):
    """
    Shake-Shake-ResNet shortcut.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 **kwargs):
        super(ShakeShakeShortcut, self).__init__(**kwargs)
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=1,
                strides=strides)
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x1 = self.pool(x)
        x1 = self.conv1(x1)
        x2 = F.slice(x, begin=(None, None, None, None), end=(None, None, -1, -1))
        x2 = F.pad(x2, mode="constant", pad_width=(0, 0, 0, 0, 1, 0, 1, 0), constant_value=0)
        x2 = self.pool(x2)
        x2 = self.conv2(x2)
        x = F.concat(x1, x2, dim=1)
        x = self.bn(x)
        return x


class ShakeShakeResUnit(HybridBlock):
    """
    Shake-Shake-ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 bottleneck,
                 **kwargs):
        super(ShakeShakeResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)
        branch_class = ResBottleneck if bottleneck else ResBlock

        with self.name_scope():
            self.branch1 = branch_class(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            self.branch2 = branch_class(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_branch = ShakeShakeShortcut(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_branch(x)
        else:
            identity = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = ShakeShake()(x1, x2) + identity
        x = self.activ(x)
        return x


class CIFARShakeShakeResNet(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARShakeShakeResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(ShakeShakeResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_shakeshakeresnet_cifar(classes,
                               blocks,
                               bottleneck,
                               first_stage_channels=16,
                               model_name=None,
                               pretrained=False,
                               ctx=cpu(),
                               root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_shakeshakeresnet_cifar(classes=classes, blocks=26, bottleneck=False, first_stage_channels=32,
                                      model_name="shakeshakeresnet26_2x32d_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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
        assert (model != shakeshakeresnet20_2x16d_cifar10 or weight_count == 541082)
        assert (model != shakeshakeresnet20_2x16d_cifar100 or weight_count == 546932)
        assert (model != shakeshakeresnet20_2x16d_svhn or weight_count == 541082)
        assert (model != shakeshakeresnet26_2x32d_cifar10 or weight_count == 2923162)
        assert (model != shakeshakeresnet26_2x32d_cifar100 or weight_count == 2934772)
        assert (model != shakeshakeresnet26_2x32d_svhn or weight_count == 2923162)

        x = mx.nd.zeros((14, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (14, classes))


if __name__ == "__main__":
    _test()
