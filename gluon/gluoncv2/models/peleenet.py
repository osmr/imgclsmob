"""
    PeleeNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Pelee: A Real-Time Object Detection System on Mobile Devices,' https://arxiv.org/abs/1804.06882.
"""

__all__ = ['PeleeNet', 'peleenet']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import conv1x1_block, conv3x3_block


class PeleeBranch1(HybridBlock):
    """
    PeleeNet branch type 1 block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 strides=1,
                 bn_use_global_stats=False,
                 **kwargs):
        super(PeleeBranch1, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PeleeBranch2(HybridBlock):
    """
    PeleeNet branch type 2 block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(PeleeBranch2, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class StemBlock(HybridBlock):
    """
    PeleeNet stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(StemBlock, self).__init__(**kwargs)
        mid1_channels = out_channels // 2
        mid2_channels = out_channels * 2

        with self.name_scope():
            self.first_conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                strides=2)

            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(PeleeBranch1(
                in_channels=out_channels,
                out_channels=out_channels,
                mid_channels=mid1_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(nn.MaxPool2D(
                pool_size=2,
                strides=2,
                padding=0))

            self.last_conv = conv1x1_block(
                in_channels=mid2_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.first_conv(x)
        x = self.branches(x)
        x = self.last_conv(x)
        return x


class DenseBlock(HybridBlock):
    """
    PeleeNet dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bottleneck_size : int
        Bottleneck width.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bottleneck_size,
                 bn_use_global_stats,
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        inc_channels = (out_channels - in_channels) // 2
        mid_channels = inc_channels * bottleneck_size

        with self.name_scope():
            self.branch1 = PeleeBranch1(
                in_channels=in_channels,
                out_channels=inc_channels,
                mid_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.branch2 = PeleeBranch2(
                in_channels=in_channels,
                out_channels=inc_channels,
                mid_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = F.concat(x, x1, x2, dim=1)
        return x


class TransitionBlock(HybridBlock):
    """
    PeleeNet's transition block, like in DensNet, but with ordinary convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.pool = nn.AvgPool2D(
                pool_size=2,
                strides=2,
                padding=0)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PeleeNet(HybridBlock):
    """
    PeleeNet model from 'Pelee: A Real-Time Object Detection System on Mobile Devices,'
    https://arxiv.org/abs/1804.06882.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck_sizes : list of int
        Bottleneck sizes for each stage.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    dropout_rate : float, default 0.5
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 bottleneck_sizes,
                 bn_use_global_stats=False,
                 dropout_rate=0.5,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(PeleeNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(StemBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                bottleneck_size = bottleneck_sizes[i]
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    if i != 0:
                        stage.add(TransitionBlock(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            bn_use_global_stats=bn_use_global_stats))
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(DenseBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bottleneck_size=bottleneck_size,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dropout(rate=dropout_rate))
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_peleenet(model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    """
    Create PeleeNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    growth_rate = 32
    layers = [3, 4, 8, 6]
    bottleneck_sizes = [1, 2, 4, 4]

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1]])[1:]],
        layers,
        [[init_block_channels]])[1:]

    net = PeleeNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck_sizes=bottleneck_sizes,
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


def peleenet(**kwargs):
    """
    PeleeNet model from 'Pelee: A Real-Time Object Detection System on Mobile Devices,'
    https://arxiv.org/abs/1804.06882.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_peleenet(model_name="peleenet", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        peleenet,
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
        assert (model != peleenet or weight_count == 2802248)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
