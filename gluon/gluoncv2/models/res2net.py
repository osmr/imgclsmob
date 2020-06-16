"""
    Res2Net for ImageNet-1K, implemented in Gluon.
    Original paper: 'Res2Net: A New Multi-scale Backbone Architecture,' https://arxiv.org/abs/1904.01169.
"""

__all__ = ['Res2Net', 'res2net50_w14_s8', 'res2net50_w26_s8']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import Identity
from .common import conv1x1, conv3x3, conv1x1_block
from .resnet import ResInitBlock
from .preresnet import PreResActivation


class HierarchicalConcurrent(nn.HybridSequential):
    """
    A container for hierarchical concatenation of blocks with parameters.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    multi_input : bool, default False
        Whether input is multiple.
    """
    def __init__(self,
                 axis=1,
                 multi_input=False,
                 **kwargs):
        super(HierarchicalConcurrent, self).__init__(**kwargs)
        self.axis = axis
        self.multi_input = multi_input

    def hybrid_forward(self, F, x):
        out = []
        y_prev = None
        if self.multi_input:
            xs = F.split(x, axis=self.axis, num_outputs=len(self._children.values()))
        for i, block in enumerate(self._children.values()):
            if self.multi_input:
                y = block(xs[i])
            else:
                y = block(x)
            if y_prev is not None:
                y = y + y_prev
            out.append(y)
            y_prev = y
        out = F.concat(*out, dim=self.axis)
        return out


class Res2NetUnit(HybridBlock):
    """
    Res2Net unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    width : int
        Width of filters.
    scale : int
        Number of scale.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 width,
                 scale,
                 bn_use_global_stats,
                 **kwargs):
        super(Res2NetUnit, self).__init__(**kwargs)
        self.scale = scale
        downsample = (strides != 1)
        self.resize_identity = (in_channels != out_channels) or downsample
        mid_channels = width * scale
        brn_channels = width

        with self.name_scope():
            self.reduce_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.branches = HierarchicalConcurrent(axis=1, multi_input=True, prefix="")
            if downsample:
                self.branches.add(conv1x1(
                    in_channels=brn_channels,
                    out_channels=brn_channels,
                    strides=strides))
            else:
                self.branches.add(Identity())
            for i in range(scale - 1):
                self.branches.add(conv3x3(
                    in_channels=brn_channels,
                    out_channels=brn_channels,
                    strides=strides))
            self.preactiv = PreResActivation(in_channels=mid_channels)
            self.merge_conv = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        y = self.reduce_conv(x)
        y = self.branches(y)
        y = self.preactiv(y)
        y = self.merge_conv(y)
        y = y + identity
        y = self.activ(y)
        return y


class Res2Net(HybridBlock):
    """
    Res2Net model from 'Res2Net: A New Multi-scale Backbone Architecture,' https://arxiv.org/abs/1904.01169.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    width : int
        Width of filters.
    scale : int
        Number of scale.
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
                 width,
                 scale,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(Res2Net, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(Res2NetUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            width=width,
                            scale=scale,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=7,
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


def get_res2net(blocks,
                width,
                scale,
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
                **kwargs):
    """
    Create Res2Net model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    width : int
        Width of filters.
    scale : int
        Number of scale.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    bottleneck = True

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported Res2Net with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = Res2Net(
        channels=channels,
        init_block_channels=init_block_channels,
        width=width,
        scale=scale,
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


def res2net50_w14_s8(**kwargs):
    """
    Res2Net-50 (14wx8s) model from 'Res2Net: A New Multi-scale Backbone Architecture,' https://arxiv.org/abs/1904.01169.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_res2net(blocks=50, width=14, scale=8, model_name="res2net50_w14_s8", **kwargs)


def res2net50_w26_s8(**kwargs):
    """
    Res2Net-50 (26wx8s) model from 'Res2Net: A New Multi-scale Backbone Architecture,' https://arxiv.org/abs/1904.01169.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_res2net(blocks=50, width=26, scale=8, model_name="res2net50_w14_s8", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        res2net50_w14_s8,
        res2net50_w26_s8,
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
        assert (model != res2net50_w14_s8 or weight_count == 8231732)
        assert (model != res2net50_w26_s8 or weight_count == 11432660)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
