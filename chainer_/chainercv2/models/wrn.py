"""
    WRN for ImageNet-1K, implemented in Chainer.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['WRN', 'wrn50_2']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


class WRNConv(Chain):
    """
    WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 activate):
        super(WRNConv, self).__init__()
        self.activate = activate

        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=False)
            if self.activate:
                self.activ = F.relu

    def __call__(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.activ(x)
        return x


def wrn_conv1x1(in_channels,
                out_channels,
                stride,
                activate):
    """
    1x1 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=0,
        activate=activate)


def wrn_conv3x3(in_channels,
                out_channels,
                stride,
                activate):
    """
    3x3 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=1,
        activate=activate)


class WRNBottleneck(Chain):
    """
    WRN bottleneck block for residual path in WRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 width_factor):
        super(WRNBottleneck, self).__init__()
        mid_channels = int(round(out_channels // 4 * width_factor))

        with self.init_scope():
            self.conv1 = wrn_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=1,
                activate=True)
            self.conv2 = wrn_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activate=True)
            self.conv3 = wrn_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                stride=1,
                activate=False)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class WRNUnit(Chain):
    """
    WRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 width_factor):
        super(WRNUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.body = WRNBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                width_factor=width_factor)
            if self.resize_identity:
                self.identity_conv = wrn_conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activate=False)
            self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class WRNInitBlock(Chain):
    """
    WRN specific initial block.

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
        super(WRNInitBlock, self).__init__()
        with self.init_scope():
            self.conv = WRNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=7,
                stride=2,
                pad=3,
                activate=True)
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class WRN(Chain):
    """
    WRN model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    width_factor : float
        Wide scale factor for width of layers.
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
                 width_factor,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(WRN, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", WRNInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), WRNUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                width_factor=width_factor))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
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


def get_wrn(blocks,
            width_factor,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".chainer", "models"),
            **kwargs):
    """
    Create WRN model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    width_factor : float
        Wide scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported WRN with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = WRN(
        channels=channels,
        init_block_channels=init_block_channels,
        width_factor=width_factor,
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


def wrn50_2(**kwargs):
    """
    WRN-50-2 model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn(blocks=50, width_factor=2.0, model_name="wrn50_2", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        wrn50_2,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn50_2 or weight_count == 68849128)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
