"""
    InceptionV3 for ImageNet-1K, implemented in Chainer.
    Original paper: 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.
"""

__all__ = ['InceptionV3', 'inceptionv3']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential, Concurrent


class InceptConv(Chain):
    """
    InceptionV3 specific convolution block.

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
    1x1 version of the InceptionV3 specific convolution block.

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


class MaxPoolBranch(Chain):
    """
    InceptionV3 specific max pooling branch block.
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
    InceptionV3 specific average pooling branch block.

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
                F.average_pooling_2d,
                ksize=3,
                stride=1,
                pad=1)
            self.conv = incept_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Conv1x1Branch(Chain):
    """
    InceptionV3 specific convolutional 1x1 branch block.

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


class ConvSeqBranch(Chain):
    """
    InceptionV3 specific convolutional sequence branch block.

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
    InceptionV3 specific convolutional sequence branch block with splitting by 3x3.

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
        super(ConvSeq3x3Branch, self).__init__()
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
                self.conv1x3 = InceptConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    ksize=(1, 3),
                    stride=1,
                    pad=(0, 1))
                self.conv3x1 = InceptConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
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
    InceptionV3 type Inception-A unit.

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
        super(InceptionAUnit, self).__init__()
        assert (out_channels > 224)
        pool_out_channels = out_channels - 224

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv1x1Branch(
                    in_channels=in_channels,
                    out_channels=64))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(48, 64),
                    kernel_size_list=(1, 5),
                    strides_list=(1, 1),
                    padding_list=(0, 2)))
                setattr(self.branches, "branch3", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(64, 96, 96),
                    kernel_size_list=(1, 3, 3),
                    strides_list=(1, 1, 1),
                    padding_list=(0, 1, 1)))
                setattr(self.branches, "branch4", AvgPoolBranch(
                    in_channels=in_channels,
                    out_channels=pool_out_channels))

    def __call__(self, x):
        x = self.branches(x)
        return x


class ReductionAUnit(Chain):
    """
    InceptionV3 type Reduction-A unit.

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
        super(ReductionAUnit, self).__init__()
        assert (in_channels == 288)
        assert (out_channels == 768)

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
                    out_channels_list=(64, 96, 96),
                    kernel_size_list=(1, 3, 3),
                    strides_list=(1, 1, 2),
                    padding_list=(0, 1, 0)))
                setattr(self.branches, "branch3", MaxPoolBranch())

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptionBUnit(Chain):
    """
    InceptionV3 type Inception-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of output channels in the 7x7 branches.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(InceptionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 768)

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv1x1Branch(
                    in_channels=in_channels,
                    out_channels=192))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(mid_channels, mid_channels, 192),
                    kernel_size_list=(1, (1, 7), (7, 1)),
                    strides_list=(1, 1, 1),
                    padding_list=(0, (0, 3), (3, 0))))
                setattr(self.branches, "branch3", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(mid_channels, mid_channels, mid_channels, mid_channels, 192),
                    kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
                    strides_list=(1, 1, 1, 1, 1),
                    padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3))))
                setattr(self.branches, "branch4", AvgPoolBranch(
                    in_channels=in_channels,
                    out_channels=192))

    def __call__(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(Chain):
    """
    InceptionV3 type Reduction-B unit.

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
        super(ReductionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 1280)

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(192, 320),
                    kernel_size_list=(1, 3),
                    strides_list=(1, 2),
                    padding_list=(0, 0)))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(192, 192, 192, 192),
                    kernel_size_list=(1, (1, 7), (7, 1), 3),
                    strides_list=(1, 1, 1, 2),
                    padding_list=(0, (0, 3), (3, 0), 0)))
                setattr(self.branches, "branch3", MaxPoolBranch())

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptionCUnit(Chain):
    """
    InceptionV3 type Inception-C unit.

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
        super(InceptionCUnit, self).__init__()
        assert (out_channels == 2048)

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv1x1Branch(
                    in_channels=in_channels,
                    out_channels=320))
                setattr(self.branches, "branch2", ConvSeq3x3Branch(
                    in_channels=in_channels,
                    out_channels_list=(384,),
                    kernel_size_list=(1,),
                    strides_list=(1,),
                    padding_list=(0,)))
                setattr(self.branches, "branch3", ConvSeq3x3Branch(
                    in_channels=in_channels,
                    out_channels_list=(448, 384),
                    kernel_size_list=(1, 3),
                    strides_list=(1, 1),
                    padding_list=(0, 1)))
                setattr(self.branches, "branch4", AvgPoolBranch(
                    in_channels=in_channels,
                    out_channels=192))

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(Chain):
    """
    InceptionV3 specific initial block.

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
        super(InceptInitBlock, self).__init__()
        assert (out_channels == 192)

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
            self.pool1 = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=0,
                cover_all=False)
            self.conv4 = InceptConv(
                in_channels=64,
                out_channels=80,
                ksize=1,
                stride=1,
                pad=0)
            self.conv5 = InceptConv(
                in_channels=80,
                out_channels=192,
                ksize=3,
                stride=1,
                pad=0)
            self.pool2 = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=0,
                cover_all=False)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        return x


class InceptionV3(Chain):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    b_mid_channels : list of int
        Number of middle channels for each Inception-B unit.
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
                 channels,
                 init_block_channels,
                 b_mid_channels,
                 dropout_rate=0.5,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000):
        super(InceptionV3, self).__init__()
        self.in_size = in_size
        self.classes = classes
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", InceptInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels

                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            if (j == 0) and (i != 0):
                                unit = reduction_units[i - 1]
                            else:
                                unit = normal_units[i]
                            if unit == InceptionBUnit:
                                setattr(stage, "unit{}".format(j + 1), unit(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    mid_channels=b_mid_channels[j - 1]))
                            else:
                                setattr(stage, "unit{}".format(j + 1), unit(
                                    in_channels=in_channels,
                                    out_channels=out_channels))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)

                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
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


def get_inceptionv3(model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".chainer", "models"),
                    **kwargs):
    """
    Create InceptionV3 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 192
    channels = [[256, 288, 288],
                [768, 768, 768, 768, 768],
                [1280, 2048, 2048]]
    b_mid_channels = [128, 160, 160, 192]

    net = InceptionV3(
        channels=channels,
        init_block_channels=init_block_channels,
        b_mid_channels=b_mid_channels,
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


def inceptionv3(**kwargs):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv3(model_name="inceptionv3", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        inceptionv3,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionv3 or weight_count == 23834568)

        x = np.zeros((1, 3, 299, 299), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
