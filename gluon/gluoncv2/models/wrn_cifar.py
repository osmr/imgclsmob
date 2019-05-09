"""
    WRN for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['CIFARWRN', 'wrn16_10_cifar10', 'wrn16_10_cifar100', 'wrn16_10_svhn', 'wrn28_10_cifar10',
           'wrn28_10_cifar100', 'wrn28_10_svhn', 'wrn40_8_cifar10', 'wrn40_8_cifar100', 'wrn40_8_svhn']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3
from .preresnet import PreResUnit, PreResActivation


class CIFARWRN(HybridBlock):
    """
    WRN model for CIFAR from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    classes : int, default 10
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
        super(CIFARWRN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(PreResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=False,
                            conv1_stride=False))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PreResActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
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


def get_wrn_cifar(classes,
                  blocks,
                  width_factor,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_wrn_cifar(classes=classes, blocks=40, width_factor=8, model_name="wrn40_8_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
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

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
