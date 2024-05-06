"""
    SKNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.
"""

__all__ = ['SKNet', 'sknet50', 'sknet101', 'sknet152']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent
from .resnet import ResInitBlock


class SKConvBlock(HybridBlock):
    """
    SKNet specific convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple(int, int)
        Strides of the convolution.
    groups : int, default 32
        Number of groups in branches.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    num_branches : int, default 2
        Number of branches (`M` parameter in the paper).
    reduction : int, default 16
        Reduction value for intermediate channels (`r` parameter in the paper).
    min_channels : int, default 32
        Minimal number of intermediate channels (`L` parameter in the paper).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 groups=32,
                 bn_use_global_stats=False,
                 num_branches=2,
                 reduction=16,
                 min_channels=32,
                 **kwargs):
        super(SKConvBlock, self).__init__(**kwargs)
        self.num_branches = num_branches
        self.out_channels = out_channels
        mid_channels = max(in_channels // reduction, min_channels)

        with self.name_scope():
            self.branches = Concurrent(stack=True, prefix="")
            for i in range(num_branches):
                dilation = 1 + i
                self.branches.add(conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=dilation,
                    dilation=dilation,
                    groups=groups,
                    bn_use_global_stats=bn_use_global_stats))
            self.fc1 = conv1x1_block(
                in_channels=out_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.fc2 = conv1x1(
                in_channels=mid_channels,
                out_channels=(out_channels * num_branches))

    def hybrid_forward(self, F, x):
        y = self.branches(x)

        u = y.sum(axis=1)
        s = F.contrib.AdaptiveAvgPooling2D(u, output_size=1)
        z = self.fc1(s)
        w = self.fc2(z)

        w = w.reshape((0, self.num_branches, self.out_channels))
        w = F.softmax(w, axis=1)
        w = w.expand_dims(3).expand_dims(4)

        y = F.broadcast_mul(y, w)
        y = y.sum(axis=1)
        return y


class SKNetBottleneck(HybridBlock):
    """
    SKNet bottleneck block for residual path in SKNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple(int, int)
        Strides of the convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 bottleneck_factor=2,
                 **kwargs):
        super(SKNetBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = SKConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SKNetUnit(HybridBlock):
    """
    SKNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple(int, int)
        Strides of the convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 **kwargs):
        super(SKNetUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = SKNetBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
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
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class SKNet(HybridBlock):
    """
    SKNet model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters
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
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(SKNet, self).__init__(**kwargs)
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
                        stage.add(SKNetUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
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


def get_sknet(blocks,
              model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    """
    Create SKNet model with specific parameters.

    Parameters
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

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported SKNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SKNet(
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


def sknet50(**kwargs):
    """
    SKNet-50 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=50, model_name="sknet50", **kwargs)


def sknet101(**kwargs):
    """
    SKNet-101 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=101, model_name="sknet101", **kwargs)


def sknet152(**kwargs):
    """
    SKNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=152, model_name="sknet152", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        sknet50,
        sknet101,
        sknet152,
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
        assert (model != sknet50 or weight_count == 27479784)
        assert (model != sknet101 or weight_count == 48736040)
        assert (model != sknet152 or weight_count == 66295656)

        x = mx.nd.zeros((14, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (14, 1000))


if __name__ == "__main__":
    _test()
