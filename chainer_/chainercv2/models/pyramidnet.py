"""
    PyramidNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.
"""

__all__ = ['PyramidNet', 'pyramidnet101_a360', 'PyrUnit']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import pre_conv1x1_block, pre_conv3x3_block, SimpleSequential
from .preresnet import PreResActivation


class PyrBlock(Chain):
    """
    Simple PyramidNet block for residual path in PyramidNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PyrBlock, self).__init__()
        with self.init_scope():
            self.conv1 = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activate=False)
            self.conv2 = pre_conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PyrBottleneck(Chain):
    """
    PyramidNet bottleneck block for residual path in PyramidNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PyrBottleneck, self).__init__()
        mid_channels = out_channels // 4

        with self.init_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activate=False)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride)
            self.conv3 = pre_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PyrUnit(Chain):
    """
    PyramidNet unit with residual connection.

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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(PyrUnit, self).__init__()
        assert (out_channels >= in_channels)
        self.resize_identity = (stride != 1)
        if out_channels > in_channels:
            self.identity_pad_width = ((0, 0), (0, out_channels - in_channels), (0, 0), (0, 0))
        else:
            self.identity_pad_width = None

        with self.init_scope():
            if bottleneck:
                self.body = PyrBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)
            else:
                self.body = PyrBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)
            if self.resize_identity:
                self.identity_pool = partial(
                    F.average_pooling_2d,
                    ksize=2,
                    stride=stride)

    def __call__(self, x):
        identity = x
        x = self.body(x)
        x = self.bn(x)
        if self.resize_identity:
            identity = self.identity_pool(identity)
        if self.identity_pad_width is not None:
            identity = F.pad(identity, pad_width=self.identity_pad_width, mode="constant", constant_values=0)
        x = x + identity
        return x


class PyrInitBlock(Chain):
    """
    PyramidNet specific initial block.

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
        super(PyrInitBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=7,
                stride=2,
                pad=3,
                nobias=True)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)
            self.activ = F.relu
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class PyramidNet(Chain):
    """
    PyramidNet model from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
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
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(PyramidNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", PyrInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), PyrUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, 'post_activ', PreResActivation(in_channels=in_channels))
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


def get_pyramidnet(blocks,
                   alpha,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".chainer", "models"),
                   **kwargs):
    """
    Create PyramidNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    alpha : int
        PyramidNet's alpha value.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14:
        layers = [2, 2, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    growth_add = float(alpha) / float(sum(layers))
    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [[(i + 1) * growth_add + xi[-1][-1] for i in list(range(yi))]],
        layers,
        [[init_block_channels]])[1:]
    channels = [[int(round(cij)) for cij in ci] for ci in channels]

    if blocks < 50:
        bottleneck = False
    else:
        bottleneck = True
        channels = [[cij * 4 for cij in ci] for ci in channels]

    net = PyramidNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
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


def pyramidnet101_a360(**kwargs):
    """
    PyramidNet-101 model from 'Deep Pyramidal Residual Networks,' https://arxiv.org/abs/1610.02915.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_pyramidnet(blocks=101, alpha=360, model_name="pyramidnet101_a360", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        pyramidnet101_a360,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pyramidnet101_a360 or weight_count == 42455070)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
