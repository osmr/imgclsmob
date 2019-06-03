"""
    Oct-ResNet for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
    Convolution,' https://arxiv.org/abs/1904.05049.
"""

__all__ = ['CIFAROctResNet', 'octresnet20_ad2_cifar10', 'octresnet20_ad2_cifar100', 'octresnet20_ad2_svhn',
           'octresnet56_ad2_cifar10', 'octresnet56_ad2_cifar100', 'octresnet56_ad2_svhn']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3_block, DualPathSequential
from .octresnet import OctResUnit


class CIFAROctResNet(HybridBlock):
    """
    Oct-ResNet model for CIFAR from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with
    Octave Convolution,' https://arxiv.org/abs/1904.05049.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    oct_alpha : float, default 0.5
        Octave alpha coefficient.
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
                 oct_alpha=0.5,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFAROctResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=1,
                prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        if (i == 0) and (j == 0):
                            oct_mode = "first"
                        elif (i == len(channels) - 1) and (j == 0):
                            oct_mode = "last"
                        elif (i == len(channels) - 1) and (j != 0):
                            oct_mode = "std"
                        else:
                            oct_mode = "norm"
                        stage.add(OctResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            oct_alpha=oct_alpha,
                            oct_mode=oct_mode,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=False))
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


def get_octresnet_cifar(classes,
                        blocks,
                        bottleneck,
                        oct_alpha=0.5,
                        model_name=None,
                        pretrained=False,
                        ctx=cpu(),
                        root=os.path.join("~", ".mxnet", "models"),
                        **kwargs):
    """
    Create Oct-ResNet model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    oct_alpha : float, default 0.5
        Octave alpha coefficient.
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

    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = CIFAROctResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        oct_alpha=oct_alpha,
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


def octresnet20_ad2_cifar10(classes=10, **kwargs):
    """
    Oct-ResNet-20 (alpha=1/2) model for CIFAR-10 from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional
    Neural Networks with Octave Convolution,' https://arxiv.org/abs/1904.05049.

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
    return get_octresnet_cifar(classes=classes, blocks=20, bottleneck=False, oct_alpha=0.5,
                               model_name="octresnet20_ad2_cifar10", **kwargs)


def octresnet20_ad2_cifar100(classes=100, **kwargs):
    """
    Oct-ResNet-20 (alpha=1/2) model for CIFAR-100 from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional
    Neural Networks with Octave Convolution,' https://arxiv.org/abs/1904.05049.

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
    return get_octresnet_cifar(classes=classes, blocks=20, bottleneck=False, oct_alpha=0.5,
                               model_name="octresnet20_ad2_cifar100", **kwargs)


def octresnet20_ad2_svhn(classes=10, **kwargs):
    """
    Oct-ResNet-20 (alpha=1/2) model for SVHN from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional
    Neural Networks with Octave Convolution,' https://arxiv.org/abs/1904.05049.

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
    return get_octresnet_cifar(classes=classes, blocks=20, bottleneck=False, oct_alpha=0.5,
                               model_name="octresnet20_ad2_svhn", **kwargs)


def octresnet56_ad2_cifar10(classes=10, **kwargs):
    """
    Oct-ResNet-56 (alpha=1/2) model for CIFAR-10 from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional
    Neural Networks with Octave Convolution,' https://arxiv.org/abs/1904.05049.

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
    return get_octresnet_cifar(classes=classes, blocks=56, bottleneck=False, oct_alpha=0.5,
                               model_name="octresnet56_ad2_cifar10", **kwargs)


def octresnet56_ad2_cifar100(classes=100, **kwargs):
    """
    Oct-ResNet-56 (alpha=1/2) model for CIFAR-100 from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional
    Neural Networks with Octave Convolution,' https://arxiv.org/abs/1904.05049.

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
    return get_octresnet_cifar(classes=classes, blocks=56, bottleneck=False, oct_alpha=0.5,
                               model_name="octresnet56_ad2_cifar100", **kwargs)


def octresnet56_ad2_svhn(classes=10, **kwargs):
    """
    Oct-ResNet-56 (alpha=1/2) model for SVHN from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional
    Neural Networks with Octave Convolution,' https://arxiv.org/abs/1904.05049.

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
    return get_octresnet_cifar(classes=classes, blocks=56, bottleneck=False, oct_alpha=0.5,
                               model_name="octresnet56_ad2_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (octresnet20_ad2_cifar10, 10),
        (octresnet20_ad2_cifar100, 100),
        (octresnet20_ad2_svhn, 10),
        (octresnet56_ad2_cifar10, 10),
        (octresnet56_ad2_cifar100, 100),
        (octresnet56_ad2_svhn, 10),
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
        assert (model != octresnet20_ad2_cifar10 or weight_count == 272762)
        assert (model != octresnet20_ad2_cifar100 or weight_count == 278612)
        assert (model != octresnet20_ad2_svhn or weight_count == 272762)
        assert (model != octresnet56_ad2_cifar10 or weight_count == 856058)
        assert (model != octresnet56_ad2_cifar100 or weight_count == 861908)
        assert (model != octresnet56_ad2_svhn or weight_count == 856058)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
