"""
    ResNeXt for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
"""

__all__ = ['CIFARResNeXt', 'resnext20_4x16d_cifar10', 'resnext20_4x16d_cifar100', 'resnext20_4x16d_svhn',
           'resnext20_8x8d_cifar10', 'resnext20_8x8d_cifar100', 'resnext20_8x8d_svhn',
           'resnext20_16x4d_cifar10', 'resnext20_16x4d_cifar100', 'resnext20_16x4d_svhn',
           'resnext20_16x8d_cifar10', 'resnext20_16x8d_cifar100', 'resnext20_16x8d_svhn',
           'resnext20_32x2d_cifar10', 'resnext20_32x2d_cifar100', 'resnext20_32x2d_svhn',
           'resnext20_32x4d_cifar10', 'resnext20_32x4d_cifar100', 'resnext20_32x4d_svhn',
           'resnext20_64x1d_cifar10', 'resnext20_64x1d_cifar100', 'resnext20_64x1d_svhn',
           'resnext29_32x4d_cifar10', 'resnext29_32x4d_cifar100', 'resnext29_32x4d_svhn',
           'resnext29_16x64d_cifar10', 'resnext29_16x64d_cifar100', 'resnext29_16x64d_svhn']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3_block
from .resnext import ResNeXtUnit


class CIFARResNeXt(HybridBlock):
    """
    ResNeXt model for CIFAR from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
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
                 cardinality,
                 bottleneck_width,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARResNeXt, self).__init__(**kwargs)
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
                        stage.add(ResNeXtUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            cardinality=cardinality,
                            bottleneck_width=bottleneck_width,
                            bn_use_global_stats=bn_use_global_stats))
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


def get_resnext_cifar(classes,
                      blocks,
                      cardinality,
                      bottleneck_width,
                      model_name=None,
                      pretrained=False,
                      ctx=cpu(),
                      root=os.path.join("~", ".mxnet", "models"),
                      **kwargs):
    """
    ResNeXt model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    assert (blocks - 2) % 9 == 0
    layers = [(blocks - 2) // 9] * 3
    channels_per_layers = [256, 512, 1024]
    init_block_channels = 64

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CIFARResNeXt(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
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


def resnext20_4x16d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-20 (4x16d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=4, bottleneck_width=16,
                             model_name="resnext20_4x16d_cifar10", **kwargs)


def resnext20_4x16d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-20 (4x16d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=4, bottleneck_width=16,
                             model_name="resnext20_4x16d_cifar100", **kwargs)


def resnext20_4x16d_svhn(classes=10, **kwargs):
    """
    ResNeXt-20 (4x16d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=4, bottleneck_width=16,
                             model_name="resnext20_4x16d_svhn", **kwargs)


def resnext20_8x8d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-20 (8x8d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=8, bottleneck_width=8,
                             model_name="resnext20_8x8d_cifar10", **kwargs)


def resnext20_8x8d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-20 (8x8d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=8, bottleneck_width=8,
                             model_name="resnext20_8x8d_cifar100", **kwargs)


def resnext20_8x8d_svhn(classes=10, **kwargs):
    """
    ResNeXt-20 (8x8d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=8, bottleneck_width=8,
                             model_name="resnext20_8x8d_svhn", **kwargs)


def resnext20_16x4d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-20 (16x4d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=16, bottleneck_width=4,
                             model_name="resnext20_16x4d_cifar10", **kwargs)


def resnext20_16x4d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-20 (16x4d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=16, bottleneck_width=4,
                             model_name="resnext20_16x4d_cifar100", **kwargs)


def resnext20_16x4d_svhn(classes=10, **kwargs):
    """
    ResNeXt-20 (16x4d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=16, bottleneck_width=4,
                             model_name="resnext20_16x4d_svhn", **kwargs)


def resnext20_16x8d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-20 (16x8d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=16, bottleneck_width=8,
                             model_name="resnext20_16x8d_cifar10", **kwargs)


def resnext20_16x8d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-20 (16x8d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=16, bottleneck_width=8,
                             model_name="resnext20_16x8d_cifar100", **kwargs)


def resnext20_16x8d_svhn(classes=10, **kwargs):
    """
    ResNeXt-20 (16x8d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=16, bottleneck_width=8,
                             model_name="resnext20_16x8d_svhn", **kwargs)


def resnext20_32x2d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-20 (32x2d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=32, bottleneck_width=2,
                             model_name="resnext20_32x2d_cifar10", **kwargs)


def resnext20_32x2d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-20 (32x2d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=32, bottleneck_width=2,
                             model_name="resnext20_32x2d_cifar100", **kwargs)


def resnext20_32x2d_svhn(classes=10, **kwargs):
    """
    ResNeXt-20 (32x2d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=32, bottleneck_width=2,
                             model_name="resnext20_32x2d_svhn", **kwargs)


def resnext20_32x4d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-20 (32x4d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=32, bottleneck_width=4,
                             model_name="resnext20_32x4d_cifar10", **kwargs)


def resnext20_32x4d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-20 (32x4d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=32, bottleneck_width=4,
                             model_name="resnext20_32x4d_cifar100", **kwargs)


def resnext20_32x4d_svhn(classes=10, **kwargs):
    """
    ResNeXt-20 (32x4d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=32, bottleneck_width=4,
                             model_name="resnext20_32x4d_svhn", **kwargs)


def resnext20_64x1d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-20 (64x1d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=64, bottleneck_width=1,
                             model_name="resnext20_64x1d_cifar10", **kwargs)


def resnext20_64x1d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-20 (64x1d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=64, bottleneck_width=1,
                             model_name="resnext20_64x1d_cifar100", **kwargs)


def resnext20_64x1d_svhn(classes=10, **kwargs):
    """
    ResNeXt-20 (64x1d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=20, cardinality=64, bottleneck_width=1,
                             model_name="resnext20_64x1d_svhn", **kwargs)


def resnext29_32x4d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-29 (32x4d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=29, cardinality=32, bottleneck_width=4,
                             model_name="resnext29_32x4d_cifar10", **kwargs)


def resnext29_32x4d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-29 (32x4d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=29, cardinality=32, bottleneck_width=4,
                             model_name="resnext29_32x4d_cifar100", **kwargs)


def resnext29_32x4d_svhn(classes=10, **kwargs):
    """
    ResNeXt-29 (32x4d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=29, cardinality=32, bottleneck_width=4,
                             model_name="resnext29_32x4d_svhn", **kwargs)


def resnext29_16x64d_cifar10(classes=10, **kwargs):
    """
    ResNeXt-29 (16x64d) model for CIFAR-10 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=29, cardinality=16, bottleneck_width=64,
                             model_name="resnext29_16x64d_cifar10", **kwargs)


def resnext29_16x64d_cifar100(classes=100, **kwargs):
    """
    ResNeXt-29 (16x64d) model for CIFAR-100 from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=29, cardinality=16, bottleneck_width=64,
                             model_name="resnext29_16x64d_cifar100", **kwargs)


def resnext29_16x64d_svhn(classes=10, **kwargs):
    """
    ResNeXt-29 (16x64d) model for SVHN from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext_cifar(classes=classes, blocks=29, cardinality=16, bottleneck_width=64,
                             model_name="resnext29_16x64d_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (resnext20_4x16d_cifar10, 10),
        (resnext20_4x16d_cifar100, 100),
        (resnext20_4x16d_svhn, 10),
        (resnext20_8x8d_cifar10, 10),
        (resnext20_8x8d_cifar100, 100),
        (resnext20_8x8d_svhn, 10),
        (resnext20_16x4d_cifar10, 10),
        (resnext20_16x4d_cifar100, 100),
        (resnext20_16x4d_svhn, 10),
        (resnext20_16x8d_cifar10, 10),
        (resnext20_16x8d_cifar100, 100),
        (resnext20_16x8d_svhn, 10),
        (resnext20_32x2d_cifar10, 10),
        (resnext20_32x2d_cifar100, 100),
        (resnext20_32x2d_svhn, 10),
        (resnext20_32x4d_cifar10, 10),
        (resnext20_32x4d_cifar100, 100),
        (resnext20_32x4d_svhn, 10),
        (resnext20_64x1d_cifar10, 10),
        (resnext20_64x1d_cifar100, 100),
        (resnext20_64x1d_svhn, 10),
        (resnext29_32x4d_cifar10, 10),
        (resnext29_32x4d_cifar100, 100),
        (resnext29_32x4d_svhn, 10),
        (resnext29_16x64d_cifar10, 10),
        (resnext29_16x64d_cifar100, 100),
        (resnext29_16x64d_svhn, 10),
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
        assert (model != resnext20_4x16d_cifar10 or weight_count == 2285386)
        assert (model != resnext20_4x16d_cifar100 or weight_count == 2377636)
        assert (model != resnext20_4x16d_svhn or weight_count == 2285386)
        assert (model != resnext20_8x8d_cifar10 or weight_count == 2091850)
        assert (model != resnext20_8x8d_cifar100 or weight_count == 2184100)
        assert (model != resnext20_8x8d_svhn or weight_count == 2091850)
        assert (model != resnext20_16x4d_cifar10 or weight_count == 1995082)
        assert (model != resnext20_16x4d_cifar100 or weight_count == 2087332)
        assert (model != resnext20_16x4d_svhn or weight_count == 1995082)
        assert (model != resnext20_16x8d_cifar10 or weight_count == 3489098)
        assert (model != resnext20_16x8d_cifar100 or weight_count == 3581348)
        assert (model != resnext20_16x8d_svhn or weight_count == 3489098)
        assert (model != resnext20_32x2d_cifar10 or weight_count == 1946698)
        assert (model != resnext20_32x2d_cifar100 or weight_count == 2038948)
        assert (model != resnext20_32x2d_svhn or weight_count == 1946698)
        assert (model != resnext20_32x4d_cifar10 or weight_count == 3295562)
        assert (model != resnext20_32x4d_cifar100 or weight_count == 3387812)
        assert (model != resnext20_32x4d_svhn or weight_count == 3295562)
        assert (model != resnext20_64x1d_cifar10 or weight_count == 1922506)
        assert (model != resnext20_64x1d_cifar100 or weight_count == 2014756)
        assert (model != resnext20_64x1d_svhn or weight_count == 1922506)
        assert (model != resnext29_32x4d_cifar10 or weight_count == 4775754)
        assert (model != resnext29_32x4d_cifar100 or weight_count == 4868004)
        assert (model != resnext29_32x4d_svhn or weight_count == 4775754)
        assert (model != resnext29_16x64d_cifar10 or weight_count == 68155210)
        assert (model != resnext29_16x64d_cifar100 or weight_count == 68247460)
        assert (model != resnext29_16x64d_svhn or weight_count == 68155210)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
