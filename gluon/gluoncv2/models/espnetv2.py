"""
    ESPNetv2 for ImageNet-1K, implemented in Gluon.
    Original paper: 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.
"""

__all__ = ['ESPNetv2', 'espnetv2_wd2', 'espnetv2_w1', 'espnetv2_w5d4', 'espnetv2_w3d2', 'espnetv2_w2']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import PReLU2, conv3x3, conv1x1_block, conv3x3_block, DualPathSequential


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
            self.activ = PReLU2(in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ShortcutBlock(HybridBlock):
    """
    ESPNetv2 shortcut block.

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
        super(ShortcutBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=(lambda: PReLU2(in_channels)))
            self.conv2 = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HierarchicalConcurrent(nn.HybridSequential):
    """
    A container for hierarchical concatenation of blocks with parameters.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self,
                 axis=1,
                 **kwargs):
        super(HierarchicalConcurrent, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x):
        out = []
        y_prev = None
        for block in self._children.values():
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
                bn_use_global_stats=bn_use_global_stats,
                activation=(lambda: PReLU2(mid_channels)))

            self.branches = HierarchicalConcurrent(axis=1, prefix="")
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
                activation=None)
            self.preactiv = PreActivation(in_channels=out_channels)
            if not self.downsample:
                self.activ = PReLU2(out_channels)

    def hybrid_forward(self, F, x, x0):
        y = self.reduce_conv(x)
        y = self.branches(y)
        y = self.preactiv(y)
        y = self.merge_conv(y)
        if not self.downsample:
            y = y + x
            y = self.activ(y)
        return y, x0


class DownsampleBlock(HybridBlock):
    """
    ESPNetv2 downsample block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    x0_channels : int
        Number of input channels for shortcut.
    dilations : list of int
        Dilation values for branches in EESP block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 x0_channels,
                 dilations,
                 bn_use_global_stats,
                 **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)
        inc_channels = out_channels - in_channels

        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=3,
                strides=2,
                padding=1)
            self.eesp = ESPBlock(
                in_channels=in_channels,
                out_channels=inc_channels,
                strides=2,
                dilations=dilations,
                bn_use_global_stats=bn_use_global_stats)
            self.shortcut_block = ShortcutBlock(
                in_channels=x0_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.activ = PReLU2(out_channels)

    def hybrid_forward(self, F, x, x0):
        y1 = self.pool(x)
        y2, _ = self.eesp(x, None)
        x = F.concat(y1, y2, dim=1)
        x0 = self.pool(x0)
        y3 = self.shortcut_block(x0)
        x = x + y3
        x = self.activ(x)
        return x, x0


class ESPInitBlock(HybridBlock):
    """
    ESPNetv2 initial block.

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
        super(ESPInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                activation=(lambda: PReLU2(out_channels)))
            self.pool = nn.AvgPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x, x0):
        x = self.conv(x)
        x0 = self.pool(x0)
        return x, x0


class ESPFinalBlock(HybridBlock):
    """
    ESPNetv2 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    final_groups : int
        Number of groups in the last convolution layer.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 final_groups,
                 bn_use_global_stats,
                 **kwargs):
        super(ESPFinalBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=in_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=(lambda: PReLU2(in_channels)))
            self.conv2 = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=final_groups,
                bn_use_global_stats=bn_use_global_stats,
                activation=(lambda: PReLU2(out_channels)))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ESPNetv2(HybridBlock):
    """
    ESPNetv2 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    final_block_groups : int
        Number of groups for the final unit.
    dilations : list of list of list of int
        Dilation values for branches in each unit.
    dropout_rate : float, default 0.2
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 final_block_channels,
                 final_block_groups,
                 dilations,
                 dropout_rate=0.2,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(ESPNetv2, self).__init__()
        self.in_size = in_size
        self.classes = classes
        x0_channels = in_channels

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=0,
                last_ordinals=2,
                prefix="")
            self.features.add(ESPInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(prefix="stage{}_".format(i + 1))
                for j, out_channels in enumerate(channels_per_stage):
                    if j == 0:
                        unit = DownsampleBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            x0_channels=x0_channels,
                            dilations=dilations[i][j],
                            bn_use_global_stats=bn_use_global_stats)
                    else:
                        unit = ESPBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=1,
                            dilations=dilations[i][j],
                            bn_use_global_stats=bn_use_global_stats)
                    stage.add(unit)
                    in_channels = out_channels
                self.features.add(stage)
            self.features.add(ESPFinalBlock(
                in_channels=in_channels,
                out_channels=final_block_channels,
                final_groups=final_block_groups,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = final_block_channels
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
        x = self.features(x, x)
        x = self.output(x)
        return x


def get_espnetv2(width_scale,
                 model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    """
    Create ESPNetv2 model with specific parameters.

    Parameters:
    ----------
    width_scale : float
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
    assert (width_scale <= 2.0)

    branches = 4
    layers = [1, 4, 8, 4]

    max_dilation_list = [6, 5, 4, 3, 2]
    max_dilations = [[max_dilation_list[i]] + [max_dilation_list[i + 1]] * (li - 1) for (i, li) in enumerate(layers)]
    dilations = [[sorted([k + 1 if k < dij else 1 for k in range(branches)]) for dij in di] for di in max_dilations]

    base_channels = 32
    weighed_base_channels = math.ceil(float(math.floor(base_channels * width_scale)) / branches) * branches
    channels_per_layers = [weighed_base_channels * pow(2, i + 1) for i in range(len(layers))]

    init_block_channels = base_channels if weighed_base_channels > base_channels else weighed_base_channels
    final_block_channels = 1024 if width_scale <= 1.5 else 1280

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ESPNetv2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        final_block_groups=branches,
        dilations=dilations,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ignore_extra=True,
            ctx=ctx)

    return net


def espnetv2_wd2(**kwargs):
    """
    ESPNetv2 x0.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=0.5, model_name="espnetv2_wd2", **kwargs)


def espnetv2_w1(**kwargs):
    """
    ESPNetv2 x1.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.0, model_name="espnetv2_w1", **kwargs)


def espnetv2_w5d4(**kwargs):
    """
    ESPNetv2 x1.25 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.25, model_name="espnetv2_w5d4", **kwargs)


def espnetv2_w3d2(**kwargs):
    """
    ESPNetv2 x1.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.5, model_name="espnetv2_w3d2", **kwargs)


def espnetv2_w2(**kwargs):
    """
    ESPNetv2 x2.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=2.0, model_name="espnetv2_w2", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        espnetv2_wd2,
        espnetv2_w1,
        espnetv2_w5d4,
        espnetv2_w3d2,
        espnetv2_w2,
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
        assert (model != espnetv2_wd2 or weight_count == 1241092)
        assert (model != espnetv2_w1 or weight_count == 1669592)
        assert (model != espnetv2_w5d4 or weight_count == 1964832)
        assert (model != espnetv2_w3d2 or weight_count == 2314120)
        assert (model != espnetv2_w2 or weight_count == 3497144)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
