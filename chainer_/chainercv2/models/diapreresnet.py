"""
    DIA-PreResNet for ImageNet-1K, implemented in Chainer.
    Original papers: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['DIAPreResNet', 'diapreresnet10', 'diapreresnet12', 'diapreresnet14', 'diapreresnetbc14b', 'diapreresnet16',
           'diapreresnet18', 'diapreresnet26', 'diapreresnetbc26b', 'diapreresnet34', 'diapreresnetbc38b',
           'diapreresnet50', 'diapreresnet50b', 'diapreresnet101', 'diapreresnet101b', 'diapreresnet152',
           'diapreresnet152b', 'diapreresnet200', 'diapreresnet200b', 'diapreresnet269b', 'DIAPreResUnit']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, DualPathSequential, SimpleSequential
from .preresnet import PreResBlock, PreResBottleneck, PreResInitBlock, PreResActivation
from .diaresnet import DIAAttention


class DIAPreResUnit(Chain):
    """
    DIA-PreResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    attention : nn.Module, default None
        Attention module.
    hold_attention : bool, default False
        Whether hold attention module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 conv1_stride,
                 attention=None,
                 hold_attention=True):
        super(DIAPreResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            if bottleneck:
                self.body = PreResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv1_stride=conv1_stride)
            else:
                self.body = PreResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)
            if self.resize_identity:
                self.identity_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)
            if hold_attention:
                self.attention = attention
        if not hold_attention:
            self.attention = attention

    def __call__(self, x, hc=None):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x, hc = self.attention(x, hc)
        x = x + identity
        return x, hc


class DIAPreResNet(Chain):
    """
    DIA-PreResNet model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
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
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(DIAPreResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", PreResInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = DualPathSequential(return_two=False)
                    attention = DIAAttention(
                        in_x_features=channels_per_stage[0],
                        in_h_features=channels_per_stage[0])
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), DIAPreResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=conv1_stride,
                                attention=attention,
                                hold_attention=(j == 0)))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(
                    in_channels=in_channels))
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_diapreresnet(blocks,
                     bottleneck=None,
                     conv1_stride=True,
                     width_scale=1.0,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".chainer", "models"),
                     **kwargs):
    """
    Create DIA-PreResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    elif blocks == 269:
        layers = [3, 30, 48, 8]
    else:
        raise ValueError("Unsupported DIA-PreResNet with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

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

    net = DIAPreResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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


def diapreresnet10(**kwargs):
    """
    DIA-PreResNet-10 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=10, model_name="diapreresnet10", **kwargs)


def diapreresnet12(**kwargs):
    """
    DIA-PreResNet-12 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=12, model_name="diapreresnet12", **kwargs)


def diapreresnet14(**kwargs):
    """
    DIA-PreResNet-14 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=14, model_name="diapreresnet14", **kwargs)


def diapreresnetbc14b(**kwargs):
    """
    DIA-PreResNet-BC-14b model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=14, bottleneck=True, conv1_stride=False, model_name="diapreresnetbc14b", **kwargs)


def diapreresnet16(**kwargs):
    """
    DIA-PreResNet-16 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=16, model_name="diapreresnet16", **kwargs)


def diapreresnet18(**kwargs):
    """
    DIA-PreResNet-18 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=18, model_name="diapreresnet18", **kwargs)


def diapreresnet26(**kwargs):
    """
    DIA-PreResNet-26 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=26, bottleneck=False, model_name="diapreresnet26", **kwargs)


def diapreresnetbc26b(**kwargs):
    """
    DIA-PreResNet-BC-26b model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=26, bottleneck=True, conv1_stride=False, model_name="diapreresnetbc26b", **kwargs)


def diapreresnet34(**kwargs):
    """
    DIA-PreResNet-34 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=34, model_name="diapreresnet34", **kwargs)


def diapreresnetbc38b(**kwargs):
    """
    DIA-PreResNet-BC-38b model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=38, bottleneck=True, conv1_stride=False, model_name="diapreresnetbc38b", **kwargs)


def diapreresnet50(**kwargs):
    """
    DIA-PreResNet-50 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=50, model_name="diapreresnet50", **kwargs)


def diapreresnet50b(**kwargs):
    """
    DIA-PreResNet-50 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=50, conv1_stride=False, model_name="diapreresnet50b", **kwargs)


def diapreresnet101(**kwargs):
    """
    DIA-PreResNet-101 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=101, model_name="diapreresnet101", **kwargs)


def diapreresnet101b(**kwargs):
    """
    DIA-PreResNet-101 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=101, conv1_stride=False, model_name="diapreresnet101b", **kwargs)


def diapreresnet152(**kwargs):
    """
    DIA-PreResNet-152 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=152, model_name="diapreresnet152", **kwargs)


def diapreresnet152b(**kwargs):
    """
    DIA-PreResNet-152 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=152, conv1_stride=False, model_name="diapreresnet152b", **kwargs)


def diapreresnet200(**kwargs):
    """
    DIA-PreResNet-200 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=200, model_name="diapreresnet200", **kwargs)


def diapreresnet200b(**kwargs):
    """
    DIA-PreResNet-200 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=200, conv1_stride=False, model_name="diapreresnet200b", **kwargs)


def diapreresnet269b(**kwargs):
    """
    DIA-PreResNet-269 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diapreresnet(blocks=269, conv1_stride=False, model_name="diapreresnet269b", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        diapreresnet10,
        diapreresnet12,
        diapreresnet14,
        diapreresnetbc14b,
        diapreresnet16,
        diapreresnet18,
        diapreresnet26,
        diapreresnetbc26b,
        diapreresnet34,
        diapreresnetbc38b,
        diapreresnet50,
        diapreresnet50b,
        diapreresnet101,
        diapreresnet101b,
        diapreresnet152,
        diapreresnet152b,
        diapreresnet200,
        diapreresnet200b,
        diapreresnet269b,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.to_gpu()
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diapreresnet10 or weight_count == 6295688)
        assert (model != diapreresnet12 or weight_count == 6369672)
        assert (model != diapreresnet14 or weight_count == 6665096)
        assert (model != diapreresnetbc14b or weight_count == 24016424)
        assert (model != diapreresnet16 or weight_count == 7845768)
        assert (model != diapreresnet18 or weight_count == 12566408)
        assert (model != diapreresnet26 or weight_count == 18837128)
        assert (model != diapreresnetbc26b or weight_count == 29946664)
        assert (model != diapreresnet34 or weight_count == 22674568)
        assert (model != diapreresnetbc38b or weight_count == 35876904)
        assert (model != diapreresnet50 or weight_count == 39508520)
        assert (model != diapreresnet50b or weight_count == 39508520)
        assert (model != diapreresnet101 or weight_count == 58500648)
        assert (model != diapreresnet101b or weight_count == 58500648)
        assert (model != diapreresnet152 or weight_count == 74144296)
        assert (model != diapreresnet152b or weight_count == 74144296)
        assert (model != diapreresnet200 or weight_count == 78625320)
        assert (model != diapreresnet200b or weight_count == 78625320)
        assert (model != diapreresnet269b or weight_count == 116024872)

        x = np.zeros((1, 3, 224, 224), np.float32)
        # import cupy
        # x = cupy.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
