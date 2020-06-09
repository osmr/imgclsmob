"""
    ESPNetv2 for ImageNet-1K, implemented in Chainer.
    Original paper: 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.
"""

__all__ = ['ESPNetv2', 'espnetv2_wd2', 'espnetv2_w1', 'espnetv2_w5d4', 'espnetv2_w3d2', 'espnetv2_w2']

import os
import math
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3, conv1x1_block, conv3x3_block, DualPathSequential, SimpleSequential


class PreActivation(Chain):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PreActivation, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5)
            self.activ = L.PReLU(shape=(in_channels,))

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ShortcutBlock(Chain):
    """
    ESPNetv2 shortcut block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ShortcutBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                activation=(lambda: L.PReLU(shape=(in_channels,))))
            self.conv2 = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HierarchicalConcurrent(SimpleSequential):
    """
    A container for hierarchical concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=1):
        super(HierarchicalConcurrent, self).__init__()
        self.axis = axis

    def __call__(self, x):
        out = []
        y_prev = None
        for name in self.layer_names:
            y = self[name](x,)
            if y_prev is not None:
                y += y_prev
            out.append(y)
            y_prev = y
        out = F.concat(tuple(out), axis=self.axis)
        return out


class ESPBlock(Chain):
    """
    ESPNetv2 block (so-called EESP block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the branch convolution layers.
    dilates : list of int
        Dilation values for branches.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilates):
        super(ESPBlock, self).__init__()
        num_branches = len(dilates)
        assert (out_channels % num_branches == 0)
        self.downsample = (stride != 1)
        mid_channels = out_channels // num_branches

        with self.init_scope():
            self.reduce_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=num_branches,
                activation=(lambda: L.PReLU(shape=(mid_channels,))))

            self.branches = HierarchicalConcurrent()
            with self.branches.init_scope():
                for i in range(num_branches):
                    setattr(self.branches, "branch{}".format(i + 1), conv3x3(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        stride=stride,
                        pad=dilates[i],
                        dilate=dilates[i],
                        groups=mid_channels))

            self.merge_conv = conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels,
                groups=num_branches,
                activation=None)
            self.preactiv = PreActivation(in_channels=out_channels)
            if not self.downsample:
                self.activ = L.PReLU(shape=(out_channels,))

    def __call__(self, x, x0):
        y = self.reduce_conv(x)
        y = self.branches(y)
        y = self.preactiv(y)
        y = self.merge_conv(y)
        if not self.downsample:
            y = y + x
            y = self.activ(y)
        return y, x0


class DownsampleBlock(Chain):
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
    dilates : list of int
        Dilation values for branches in EESP block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 x0_channels,
                 dilates):
        super(DownsampleBlock, self).__init__()
        inc_channels = out_channels - in_channels

        with self.init_scope():
            self.pool = partial(
                F.average_pooling_2d,
                ksize=3,
                stride=2,
                pad=1)
            self.eesp = ESPBlock(
                in_channels=in_channels,
                out_channels=inc_channels,
                stride=2,
                dilates=dilates)
            self.shortcut_block = ShortcutBlock(
                in_channels=x0_channels,
                out_channels=out_channels)
            self.activ = L.PReLU(shape=(out_channels,))

    def __call__(self, x, x0):
        y1 = self.pool(x)
        y2, _ = self.eesp(x, None)
        x = F.concat((y1, y2), axis=1)
        x0 = self.pool(x0)
        y3 = self.shortcut_block(x0)
        x = x + y3
        x = self.activ(x)
        return x, x0


class ESPInitBlock(Chain):
    """
    ESPNetv2 initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ESPInitBlock, self).__init__()
        with self.init_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                activation=(lambda: L.PReLU(shape=(out_channels,))))
            self.pool = partial(
                F.average_pooling_2d,
                ksize=3,
                stride=2,
                pad=1)

    def __call__(self, x, x0):
        x = self.conv(x)
        x0 = self.pool(x0)
        return x, x0


class ESPFinalBlock(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 final_groups):
        super(ESPFinalBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=in_channels,
                activation=(lambda: L.PReLU(shape=(in_channels,))))
            self.conv2 = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=final_groups,
                activation=(lambda: L.PReLU(shape=(out_channels,))))

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ESPNetv2(Chain):
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
    dilates : list of list of list of int
        Dilation values for branches in each unit.
    dropout_rate : float, default 0.2
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
                 final_block_channels,
                 final_block_groups,
                 dilates,
                 dropout_rate=0.2,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(ESPNetv2, self).__init__()
        self.in_size = in_size
        self.classes = classes
        x0_channels = in_channels

        with self.init_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=0,
                last_ordinals=2)
            with self.features.init_scope():
                setattr(self.features, "init_block", ESPInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = DualPathSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            if j == 0:
                                unit = DownsampleBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    x0_channels=x0_channels,
                                    dilates=dilates[i][j])
                            else:
                                unit = ESPBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=1,
                                    dilates=dilates[i][j])
                            setattr(stage, "unit{}".format(j + 1), unit)
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_block", ESPFinalBlock(
                    in_channels=in_channels,
                    out_channels=final_block_channels,
                    final_groups=final_block_groups))
                in_channels = final_block_channels
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "dropout", partial(
                    F.dropout,
                    ratio=0.2))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x, x)
        x = self.output(x)
        return x


def get_espnetv2(width_scale,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        dilates=dilations,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def espnetv2_wd2(**kwargs):
    """
    ESPNetv2 x0.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=2.0, model_name="espnetv2_w2", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

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
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espnetv2_wd2 or weight_count == 1241092)
        assert (model != espnetv2_w1 or weight_count == 1669592)
        assert (model != espnetv2_w5d4 or weight_count == 1964832)
        assert (model != espnetv2_w3d2 or weight_count == 2314120)
        assert (model != espnetv2_w2 or weight_count == 3497144)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
