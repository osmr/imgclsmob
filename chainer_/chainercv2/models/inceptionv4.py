"""
    InceptionV4 for ImageNet-1K, implemented in Chainer.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionV4', 'inceptionv4']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential, Concurrent


class InceptConv(Chain):
    """
    InceptionV4 specific convolution block.

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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad):
        super(InceptConv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True)
            self.bn = L.BatchNormalization(
                size=out_channels,
                decay=0.1,
                eps=1e-3)
            self.activ = F.relu

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


def incept_conv1x1(in_channels,
                   out_channels):
    """
    1x1 version of the InceptionV4 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=1,
        pad=0)


def incept_conv3x3(in_channels,
                   out_channels,
                   stride,
                   padding=1):
    """
    3x3 version of the InceptionV4 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=padding)


class MaxPoolBranch(Chain):
    """
    InceptionV4 specific max pooling branch block.
    """
    def __init__(self):
        super(MaxPoolBranch, self).__init__()
        with self.init_scope():
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=0,
                cover_all=False)

    def __call__(self, x):
        x = self.pool(x)
        return x


class AvgPoolBranch(Chain):
    """
    InceptionV4 specific average pooling branch block.

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
        super(AvgPoolBranch, self).__init__()
        with self.init_scope():
            self.pool = partial(
                F.average_pooling_nd,
                ksize=3,
                stride=1,
                pad=1,
                pad_value=None)
            self.conv = incept_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Conv1x1Branch(Chain):
    """
    InceptionV4 specific convolutional 1x1 branch block.

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
        super(Conv1x1Branch, self).__init__()
        with self.init_scope():
            self.conv = incept_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.conv(x)
        return x


class Conv3x3Branch(Chain):
    """
    InceptionV4 specific convolutional 3x3 branch block.

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
        super(Conv3x3Branch, self).__init__()
        with self.init_scope():
            self.conv = incept_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                padding=0)

    def __call__(self, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(Chain):
    """
    InceptionV4 specific convolutional sequence branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list):
        super(ConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        with self.init_scope():
            self.conv_list = SimpleSequential()
            with self.conv_list.init_scope():
                for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                        out_channels_list, kernel_size_list, strides_list, padding_list)):
                    setattr(self.conv_list, "conv{}".format(i + 1), InceptConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksize=kernel_size,
                        stride=strides,
                        pad=padding))
                    in_channels = out_channels

    def __call__(self, x):
        x = self.conv_list(x)
        return x


class ConvSeq3x3Branch(Chain):
    """
    InceptionV4 specific convolutional sequence branch block with splitting by 3x3.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels_list : list of tuple of int
        List of numbers of output channels for middle layers.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list):
        super(ConvSeq3x3Branch, self).__init__()
        with self.init_scope():
            self.conv_list = SimpleSequential()
            with self.conv_list.init_scope():
                for i, (mid_channels, kernel_size, strides, padding) in enumerate(zip(
                        mid_channels_list, kernel_size_list, strides_list, padding_list)):
                    setattr(self.conv_list, "conv{}".format(i + 1), InceptConv(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        ksize=kernel_size,
                        stride=strides,
                        pad=padding))
                    in_channels = mid_channels
                self.conv1x3 = InceptConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    ksize=(1, 3),
                    stride=1,
                    pad=(0, 1))
                self.conv3x1 = InceptConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    ksize=(3, 1),
                    stride=1,
                    pad=(1, 0))

    def __call__(self, x):
        x = self.conv_list(x)
        y1 = self.conv1x3(x)
        y2 = self.conv3x1(x)
        x = F.concat((y1, y2), axis=1)
        return x


class InceptionAUnit(Chain):
    """
    InceptionV4 type Inception-A unit.
    """
    def __init__(self):
        super(InceptionAUnit, self).__init__()
        in_channels = 384

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv1x1Branch(
                    in_channels=in_channels,
                    out_channels=96))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(64, 96),
                    kernel_size_list=(1, 3),
                    strides_list=(1, 1),
                    padding_list=(0, 1)))
                setattr(self.branches, "branch3", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(64, 96, 96),
                    kernel_size_list=(1, 3, 3),
                    strides_list=(1, 1, 1),
                    padding_list=(0, 1, 1)))
                setattr(self.branches, "branch4", AvgPoolBranch(
                    in_channels=in_channels,
                    out_channels=96))

    def __call__(self, x):
        x = self.branches(x)
        return x


class ReductionAUnit(Chain):
    """
    InceptionV4 type Reduction-A unit.
    """
    def __init__(self):
        super(ReductionAUnit, self).__init__()
        in_channels = 384

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(384,),
                    kernel_size_list=(3,),
                    strides_list=(2,),
                    padding_list=(0,)))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(192, 224, 256),
                    kernel_size_list=(1, 3, 3),
                    strides_list=(1, 1, 2),
                    padding_list=(0, 1, 0)))
                setattr(self.branches, "branch3", MaxPoolBranch())

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptionBUnit(Chain):
    """
    InceptionV4 type Inception-B unit.
    """
    def __init__(self):
        super(InceptionBUnit, self).__init__()
        in_channels = 1024

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv1x1Branch(
                    in_channels=in_channels,
                    out_channels=384))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(192, 224, 256),
                    kernel_size_list=(1, (1, 7), (7, 1)),
                    strides_list=(1, 1, 1),
                    padding_list=(0, (0, 3), (3, 0))))
                setattr(self.branches, "branch3", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(192, 192, 224, 224, 256),
                    kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
                    strides_list=(1, 1, 1, 1, 1),
                    padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3))))
                setattr(self.branches, "branch4", AvgPoolBranch(
                    in_channels=in_channels,
                    out_channels=128))

    def __call__(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(Chain):
    """
    InceptionV4 type Reduction-B unit.
    """
    def __init__(self):
        super(ReductionBUnit, self).__init__()
        in_channels = 1024

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(192, 192),
                    kernel_size_list=(1, 3),
                    strides_list=(1, 2),
                    padding_list=(0, 0)))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(256, 256, 320, 320),
                    kernel_size_list=(1, (1, 7), (7, 1), 3),
                    strides_list=(1, 1, 1, 2),
                    padding_list=(0, (0, 3), (3, 0), 0)))
                setattr(self.branches, "branch3", MaxPoolBranch())

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptionCUnit(Chain):
    """
    InceptionV4 type Inception-C unit.
    """
    def __init__(self):
        super(InceptionCUnit, self).__init__()
        in_channels = 1536

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv1x1Branch(
                    in_channels=in_channels,
                    out_channels=256))
                setattr(self.branches, "branch2", ConvSeq3x3Branch(
                    in_channels=in_channels,
                    out_channels=256,
                    mid_channels_list=(384,),
                    kernel_size_list=(1,),
                    strides_list=(1,),
                    padding_list=(0,)))
                setattr(self.branches, "branch3", ConvSeq3x3Branch(
                    in_channels=in_channels,
                    out_channels=256,
                    mid_channels_list=(384, 448, 512),
                    kernel_size_list=(1, (3, 1), (1, 3)),
                    strides_list=(1, 1, 1),
                    padding_list=(0, (1, 0), (0, 1))))
                setattr(self.branches, "branch4", AvgPoolBranch(
                    in_channels=in_channels,
                    out_channels=256))

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptBlock3a(Chain):
    """
    InceptionV4 type Mixed-3a block.
    """
    def __init__(self):
        super(InceptBlock3a, self).__init__()
        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", MaxPoolBranch())
                setattr(self.branches, "branch2", Conv3x3Branch(
                    in_channels=64,
                    out_channels=96))

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptBlock4a(Chain):
    """
    InceptionV4 type Mixed-4a block.
    """
    def __init__(self):
        super(InceptBlock4a, self).__init__()
        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", ConvSeqBranch(
                    in_channels=160,
                    out_channels_list=(64, 96),
                    kernel_size_list=(1, 3),
                    strides_list=(1, 1),
                    padding_list=(0, 0)))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=160,
                    out_channels_list=(64, 64, 64, 96),
                    kernel_size_list=(1, (1, 7), (7, 1), 3),
                    strides_list=(1, 1, 1, 1),
                    padding_list=(0, (0, 3), (3, 0), 0)))

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptBlock5a(Chain):
    """
    InceptionV4 type Mixed-5a block.
    """
    def __init__(self):
        super(InceptBlock5a, self).__init__()
        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv3x3Branch(
                    in_channels=192,
                    out_channels=192))
                setattr(self.branches, "branch2", MaxPoolBranch())

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(Chain):
    """
    InceptionV4 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(InceptInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = InceptConv(
                in_channels=in_channels,
                out_channels=32,
                ksize=3,
                stride=2,
                pad=0)
            self.conv2 = InceptConv(
                in_channels=32,
                out_channels=32,
                ksize=3,
                stride=1,
                pad=0)
            self.conv3 = InceptConv(
                in_channels=32,
                out_channels=64,
                ksize=3,
                stride=1,
                pad=1)
            self.block1 = InceptBlock3a()
            self.block2 = InceptBlock4a()
            self.block3 = InceptBlock5a()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class InceptionV4(Chain):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000):
        super(InceptionV4, self).__init__()
        self.in_size = in_size
        self.classes = classes
        layers = [4, 8, 4]
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", InceptInitBlock(
                    in_channels=in_channels))

                for i, layers_per_stage in enumerate(layers):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j in range(layers_per_stage):
                            if (j == 0) and (i != 0):
                                unit = reduction_units[i - 1]
                            else:
                                unit = normal_units[i]
                            setattr(stage, "unit{}".format(j + 1), unit())
                    setattr(self.features, "stage{}".format(i + 1), stage)

                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))

            in_channels = 1536
            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                if dropout_rate > 0.0:
                    setattr(self.output, "dropout", partial(
                        F.dropout,
                        ratio=dropout_rate))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_inceptionv4(model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".chainer", "models"),
                    **kwargs):
    """
    Create InceptionV4 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    net = InceptionV4(**kwargs)

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


def inceptionv4(**kwargs):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv4(model_name="inceptionv4", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        inceptionv4,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionv4 or weight_count == 42679816)

        x = np.zeros((1, 3, 299, 299), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
