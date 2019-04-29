"""
    Res2Net for ImageNet-1K, implemented in Gluon.
    Original paper: 'Res2Net: A New Multi-scale Backbone Architecture,' https://arxiv.org/abs/1904.01169.
"""

__all__ = ['Res2Net', 'res2net50']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3, conv1x1_block, conv3x3_block
from .resnet import ResInitBlock


class PreActivation(HybridBlock):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(PreActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


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
        for i, block in enumerate(self._children.values()):
            if self.multi_input:
                y = block(x[i])
            else:
                y = block(x)
            if y_prev is not None:
                y = y + y_prev
            out.append(y)
            y_prev = y
        out = F.concat(*out, dim=self.axis)
        return out


class ESPBlock(HybridBlock):
    """
    ESPNetv2 block (so-called EESP block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    dilations : list of int
        Dilation values for branches.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 dilations,
                 bn_use_global_stats,
                 **kwargs):
        super(ESPBlock, self).__init__(**kwargs)
        num_branches = len(dilations)
        assert (out_channels % num_branches == 0)
        self.downsample = (strides != 1)
        mid_channels = out_channels // num_branches

        with self.name_scope():
            self.reduce_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=num_branches,
                bn_use_global_stats=bn_use_global_stats)

            self.branches = HierarchicalConcurrent(axis=1, multi_input=True, prefix='')
            self.branches.add(nn.Identity())
            for i in range(num_branches):
                self.branches.add(conv3x3(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    padding=dilations[i],
                    dilation=dilations[i],
                    groups=mid_channels))

            self.merge_conv = conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels,
                groups=num_branches,
                bn_use_global_stats=bn_use_global_stats,
                activation=None,
                activate=False)
            self.preactiv = PreActivation(in_channels=out_channels)
            if not self.downsample:
                self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x, x0):
        y = self.reduce_conv(x)
        y = self.branches(y)
        y = self.preactiv(y)
        y = self.merge_conv(y)
        if not self.downsample:
            y = y + x
            y = self.activ(y)
        return y, x0


class Res2NetBlock(HybridBlock):
    """
    Res2Net block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 bottleneck_factor=4,
                 **kwargs):
        super(Res2NetBlock, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                padding=padding,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None,
                activate=False)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


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
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 **kwargs):
        super(Res2NetUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = Res2NetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                padding=padding,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None,
                    activate=False)
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


class Res2Net(HybridBlock):
    """
    Res2Net model from 'Res2Net: A New Multi-scale Backbone Architecture,' https://arxiv.org/abs/1904.01169.

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
        super(Res2Net, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
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
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=7,
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


def get_res2net(blocks,
                width_scale=1.0,
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join('~', '.mxnet', 'models'),
                **kwargs):
    """
    Create Res2Net model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    width_scale : float, default 1.0
        Scale factor for width of layers.
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

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = Res2Net(
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


def res2net50(**kwargs):
    """
    Res2Net-50 model from 'Res2Net: A New Multi-scale Backbone Architecture,' https://arxiv.org/abs/1904.01169.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_res2net(blocks=50, model_name="res2net50", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        res2net50,
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
        assert (model != res2net50 or weight_count == 25557032)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
