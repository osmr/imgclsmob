"""
    CondenseNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.
"""

__all__ = ['CondenseNet', 'condensenet74_c4_g4', 'condensenet74_c8_g8']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import ChannelShuffle


class CondenseSimpleConv(HybridBlock):
    """
    CondenseNet specific simple convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups,
                 bn_use_global_stats,
                 **kwargs):
        super(CondenseSimpleConv, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def condense_simple_conv3x3(in_channels,
                            out_channels,
                            groups,
                            bn_use_global_stats):
    """
    3x3 version of the CondenseNet specific simple convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return CondenseSimpleConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1,
        groups=groups,
        bn_use_global_stats=bn_use_global_stats)


class CondenseComplexConv(HybridBlock):
    """
    CondenseNet specific complex convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups,
                 bn_use_global_stats,
                 **kwargs):
        super(CondenseComplexConv, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
                use_bias=False,
                in_channels=in_channels)
            self.c_shuffle = ChannelShuffle(
                channels=out_channels,
                groups=groups)
            self.index = self.params.get(
                "index",
                grad_req="null",
                shape=(in_channels,),
                init="zeros",
                allow_deferred_init=True,
                differentiable=False)

    def hybrid_forward(self, F, x, index):
        x = F.take(x, index, axis=1)
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        x = self.c_shuffle(x)
        return x


def condense_complex_conv1x1(in_channels,
                             out_channels,
                             groups,
                             bn_use_global_stats):
    """
    1x1 version of the CondenseNet specific complex convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return CondenseComplexConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        groups=groups,
        bn_use_global_stats=bn_use_global_stats)


class CondenseUnit(HybridBlock):
    """
    CondenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 bn_use_global_stats,
                 **kwargs):
        super(CondenseUnit, self).__init__(**kwargs)
        bottleneck_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bottleneck_size

        with self.name_scope():
            self.conv1 = condense_complex_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=groups,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = condense_simple_conv3x3(
                in_channels=mid_channels,
                out_channels=inc_channels,
                groups=groups,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.concat(identity, x, dim=1)
        return x


class TransitionBlock(HybridBlock):
    """
    CondenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.
    """
    def __init__(self,
                 **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=2,
                strides=2,
                padding=0)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        return x


class CondenseInitBlock(HybridBlock):
    """
    CondenseNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(CondenseInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=3,
                strides=2,
                padding=1,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x


class PostActivation(HybridBlock):
    """
    CondenseNet final block, which performs the same function of postactivation as in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(PostActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class CondenseDense(HybridBlock):
    """
    CondenseNet specific dense block.

    Parameters:
    ----------
    units : int
        Number of output channels.
    in_units : int
        Number of input channels.
    drop_rate : float
        Fraction of input channels for drop.
    """
    def __init__(self,
                 units,
                 in_units,
                 drop_rate=0.5,
                 **kwargs):
        super(CondenseDense, self).__init__(**kwargs)
        drop_in_units = int(in_units * drop_rate)
        with self.name_scope():
            self.dense = nn.Dense(
                units=units,
                in_units=drop_in_units)
            self.index = self.params.get(
                "index",
                grad_req="null",
                shape=(drop_in_units,),
                init="zeros",
                allow_deferred_init=True,
                differentiable=False)

    def hybrid_forward(self, F, x, index):
        x = F.take(x, index, axis=1)
        x = self.dense(x)
        return x


class CondenseNet(HybridBlock):
    """
    CondenseNet model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    groups : int
        Number of groups in convolution layers.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 groups,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(CondenseNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(CondenseInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    if i != 0:
                        stage.add(TransitionBlock())
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(CondenseUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            groups=groups,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PostActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(CondenseDense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_condensenet(num_layers,
                    groups=4,
                    model_name=None,
                    pretrained=False,
                    ctx=cpu(),
                    root=os.path.join("~", ".mxnet", "models"),
                    **kwargs):
    """
    Create CondenseNet (converted) model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    groups : int
        Number of groups in convolution layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    if num_layers == 74:
        init_block_channels = 16
        layers = [4, 6, 8, 10, 8]
        growth_rates = [8, 16, 32, 64, 128]
    else:
        raise ValueError("Unsupported CondenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(lambda xi, yi:
                      xi + [reduce(lambda xj, yj:
                                   xj + [xj[-1] + yj],
                                   [yi[1]] * yi[0],
                                   [xi[-1][-1]])[1:]],
                      zip(layers, growth_rates),
                      [[init_block_channels]])[1:]

    net = CondenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        groups=groups,
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


def condensenet74_c4_g4(**kwargs):
    """
    CondenseNet-74 (C=G=4) model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_condensenet(num_layers=74, groups=4, model_name="condensenet74_c4_g4", **kwargs)


def condensenet74_c8_g8(**kwargs):
    """
    CondenseNet-74 (C=G=8) model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_condensenet(num_layers=74, groups=8, model_name="condensenet74_c8_g8", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        condensenet74_c4_g4,
        condensenet74_c8_g8,
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
        assert (model != condensenet74_c4_g4 or weight_count == 4773944)
        assert (model != condensenet74_c8_g8 or weight_count == 2935416)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
