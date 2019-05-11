"""
    MnasNet, implemented in Chainer.
    Original paper: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,' https://arxiv.org/abs/1807.11626.
"""

__all__ = ['MnasNet', 'mnasnet']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


class ConvBlock(Chain):
    """
    Standard convolution block with Batch normalization and ReLU activation.

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
    groups : int, default 1
        Number of groups.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 groups=1,
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True,
                groups=groups)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)
            if self.activate:
                self.activ = F.relu

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  activate=True):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=1,
        pad=0,
        groups=1,
        activate=activate)


def dwconv_block(in_channels,
                 out_channels,
                 ksize,
                 stride,
                 activate=True):
    """
    Depthwise version of the standard convolution block.

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
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=ksize,
        stride=stride,
        pad=(ksize // 2),
        groups=out_channels,
        activate=activate)


class DwsConvBlock(Chain):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
        super(DwsConvBlock, self).__init__()
        with self.init_scope():
            self.dw_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                ksize=3,
                stride=1)
            self.pw_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MnasUnit(Chain):
    """
    MnasNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the second convolution layer.
    expansion_factor : int
        Factor for expansion of channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 expansion_factor):
        super(MnasUnit, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * expansion_factor

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activate=True)
            self.conv2 = dwconv_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                ksize=ksize,
                stride=stride,
                activate=True)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activate=False)

    def __call__(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class MnasInitBlock(Chain):
    """
    MnasNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of 2 int
        Numbers of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(MnasInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels_list[0],
                ksize=3,
                stride=2,
                pad=1,
                groups=1,
                activate=True)
            self.conv2 = DwsConvBlock(
                in_channels=out_channels_list[0],
                out_channels=out_channels_list[1])

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MnasNet(Chain):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Numbers of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    ksizes : list of list of int
        Number of kernel sizes for each unit.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
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
                 ksizes,
                 expansion_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(MnasNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", MnasInitBlock(
                    in_channels=in_channels,
                    out_channels_list=init_block_channels))
                in_channels = init_block_channels[-1]
                for i, channels_per_stage in enumerate(channels):
                    ksizes_per_stage = ksizes[i]
                    expansion_factors_per_stage = expansion_factors[i]
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            ksize = ksizes_per_stage[j]
                            expansion_factor = expansion_factors_per_stage[j]
                            stride = 2 if (j == 0) else 1
                            setattr(stage, "unit{}".format(j + 1), MnasUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                ksize=ksize,
                                stride=stride,
                                expansion_factor=expansion_factor))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, 'final_block', conv1x1_block(
                    in_channels=in_channels,
                    out_channels=final_block_channels,
                    activate=True))
                in_channels = final_block_channels
                setattr(self.features, 'final_pool', partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, 'flatten', partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, 'fc', L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mnasnet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".chainer", "models"),
                **kwargs):
    """
    Create MnasNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    init_block_channels = [32, 16]
    final_block_channels = 1280
    layers = [3, 3, 3, 2, 4, 1]
    downsample = [1, 1, 1, 0, 1, 0]
    channels_per_layers = [24, 40, 80, 96, 192, 320]
    expansion_factors_per_layers = [3, 3, 6, 6, 6, 6]
    ksizes_per_layers = [3, 5, 5, 3, 5, 3]
    default_kernel_size = 3

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    ksizes = reduce(lambda x, y: x + [[y[0]] + [default_kernel_size] * (y[1] - 1)] if y[2] != 0 else x[:-1] + [
        x[-1] + [y[0]] + [default_kernel_size] * (y[1] - 1)], zip(ksizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])

    net = MnasNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        ksizes=ksizes,
        expansion_factors=expansion_factors,
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


def mnasnet(**kwargs):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(model_name="mnasnet", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        mnasnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mnasnet or weight_count == 4308816)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
