"""
    InceptionV4 for ImageNet-1K, implemented in Gluon.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionV4', 'inceptionv4']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent


class InceptConv(HybridBlock):
    """
    InceptionV4 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(
                momentum=0.1,
                epsilon=1e-3,
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


def incept_conv1x1(in_channels,
                   out_channels,
                   bn_use_global_stats):
    """
    1x1 version of the InceptionV4 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        bn_use_global_stats=bn_use_global_stats)


def incept_conv3x3(in_channels,
                   out_channels,
                   stride,
                   padding=1,
                   bn_use_global_stats=False):
    """
    3x3 version of the InceptionV4 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=stride,
        padding=padding,
        bn_use_global_stats=bn_use_global_stats)


class MaxPoolBranch(HybridBlock):
    """
    InceptionV4 specific max pooling branch block.
    """
    def __init__(self,
                 **kwargs):
        super(MaxPoolBranch, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        return x


class AvgPoolBranch(HybridBlock):
    """
    InceptionV4 specific average pooling branch block.

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
        super(AvgPoolBranch, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)
            self.conv = incept_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Conv1x1Branch(HybridBlock):
    """
    InceptionV4 specific convolutional 1x1 branch block.

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
        super(Conv1x1Branch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = incept_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x


class Conv3x3Branch(HybridBlock):
    """
    InceptionV4 specific convolutional 3x3 branch block.

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
        super(Conv3x3Branch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = incept_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(HybridBlock):
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
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 bn_use_global_stats,
                 **kwargs):
        super(ConvSeqBranch, self).__init__(**kwargs)
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        with self.name_scope():
            self.conv_list = nn.HybridSequential(prefix="")
            for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                    out_channels_list, kernel_size_list, strides_list, padding_list)):
                self.conv_list.add(InceptConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = out_channels

    def hybrid_forward(self, F, x):
        x = self.conv_list(x)
        return x


class ConvSeq3x3Branch(HybridBlock):
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
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 bn_use_global_stats,
                 **kwargs):
        super(ConvSeq3x3Branch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv_list = nn.HybridSequential(prefix="")
            for i, (mid_channels, kernel_size, strides, padding) in enumerate(zip(
                    mid_channels_list, kernel_size_list, strides_list, padding_list)):
                self.conv_list.add(InceptConv(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = mid_channels
            self.conv1x3 = InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                strides=1,
                padding=(0, 1),
                bn_use_global_stats=bn_use_global_stats)
            self.conv3x1 = InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 1),
                strides=1,
                padding=(1, 0),
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv_list(x)
        y1 = self.conv1x3(x)
        y2 = self.conv3x1(x)
        x = F.concat(y1, y2, dim=1)
        return x


class InceptionAUnit(HybridBlock):
    """
    InceptionV4 type Inception-A unit.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptionAUnit, self).__init__(**kwargs)
        in_channels = 384

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=96,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(64, 96),
                kernel_size_list=(1, 3),
                strides_list=(1, 1),
                padding_list=(0, 1),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(64, 96, 96),
                kernel_size_list=(1, 3, 3),
                strides_list=(1, 1, 1),
                padding_list=(0, 1, 1),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(AvgPoolBranch(
                in_channels=in_channels,
                out_channels=96,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class ReductionAUnit(HybridBlock):
    """
    InceptionV4 type Reduction-A unit.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(ReductionAUnit, self).__init__(**kwargs)
        in_channels = 384

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(384,),
                kernel_size_list=(3,),
                strides_list=(2,),
                padding_list=(0,),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(192, 224, 256),
                kernel_size_list=(1, 3, 3),
                strides_list=(1, 1, 2),
                padding_list=(0, 1, 0),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(MaxPoolBranch())

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptionBUnit(HybridBlock):
    """
    InceptionV4 type Inception-B unit.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptionBUnit, self).__init__(**kwargs)
        in_channels = 1024

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=384,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(192, 224, 256),
                kernel_size_list=(1, (1, 7), (7, 1)),
                strides_list=(1, 1, 1),
                padding_list=(0, (0, 3), (3, 0)),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(192, 192, 224, 224, 256),
                kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
                strides_list=(1, 1, 1, 1, 1),
                padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3)),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(AvgPoolBranch(
                in_channels=in_channels,
                out_channels=128,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class ReductionBUnit(HybridBlock):
    """
    InceptionV4 type Reduction-B unit.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(ReductionBUnit, self).__init__(**kwargs)
        in_channels = 1024

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(192, 192),
                kernel_size_list=(1, 3),
                strides_list=(1, 2),
                padding_list=(0, 0),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(256, 256, 320, 320),
                kernel_size_list=(1, (1, 7), (7, 1), 3),
                strides_list=(1, 1, 1, 2),
                padding_list=(0, (0, 3), (3, 0), 0),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(MaxPoolBranch())

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptionCUnit(HybridBlock):
    """
    InceptionV4 type Inception-C unit.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptionCUnit, self).__init__(**kwargs)
        in_channels = 1536

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=256,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeq3x3Branch(
                in_channels=in_channels,
                out_channels=256,
                mid_channels_list=(384,),
                kernel_size_list=(1,),
                strides_list=(1,),
                padding_list=(0,),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeq3x3Branch(
                in_channels=in_channels,
                out_channels=256,
                mid_channels_list=(384, 448, 512),
                kernel_size_list=(1, (3, 1), (1, 3)),
                strides_list=(1, 1, 1),
                padding_list=(0, (1, 0), (0, 1)),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(AvgPoolBranch(
                in_channels=in_channels,
                out_channels=256,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptBlock3a(HybridBlock):
    """
    InceptionV4 type Mixed-3a block.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptBlock3a, self).__init__(**kwargs)
        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(MaxPoolBranch())
            self.branches.add(Conv3x3Branch(
                in_channels=64,
                out_channels=96,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptBlock4a(HybridBlock):
    """
    InceptionV4 type Mixed-4a block.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptBlock4a, self).__init__(**kwargs)
        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(ConvSeqBranch(
                in_channels=160,
                out_channels_list=(64, 96),
                kernel_size_list=(1, 3),
                strides_list=(1, 1),
                padding_list=(0, 0),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=160,
                out_channels_list=(64, 64, 64, 96),
                kernel_size_list=(1, (1, 7), (7, 1), 3),
                strides_list=(1, 1, 1, 1),
                padding_list=(0, (0, 3), (3, 0), 0),
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptBlock5a(HybridBlock):
    """
    InceptionV4 type Mixed-5a block.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptBlock5a, self).__init__(**kwargs)
        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv3x3Branch(
                in_channels=192,
                out_channels=192,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(MaxPoolBranch())

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptInitBlock(HybridBlock):
    """
    InceptionV4 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = InceptConv(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                strides=2,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = InceptConv(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                strides=1,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = InceptConv(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                strides=1,
                padding=1,
                bn_use_global_stats=bn_use_global_stats)
            self.block1 = InceptBlock3a(bn_use_global_stats=bn_use_global_stats)
            self.block2 = InceptBlock4a(bn_use_global_stats=bn_use_global_stats)
            self.block3 = InceptBlock5a(bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class InceptionV4(HybridBlock):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 dropout_rate=0.0,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 **kwargs):
        super(InceptionV4, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        layers = [4, 8, 4]
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(InceptInitBlock(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))

            for i, layers_per_stage in enumerate(layers):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j in range(layers_per_stage):
                        if (j == 0) and (i != 0):
                            unit = reduction_units[i - 1]
                        else:
                            unit = normal_units[i]
                        stage.add(unit(bn_use_global_stats=bn_use_global_stats))
                self.features.add(stage)

            self.features.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            if dropout_rate > 0.0:
                self.output.add(nn.Dropout(rate=dropout_rate))
            self.output.add(nn.Dense(
                units=classes,
                in_units=1536))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_inceptionv4(model_name=None,
                    pretrained=False,
                    ctx=cpu(),
                    root=os.path.join("~", ".mxnet", "models"),
                    **kwargs):
    """
    Create InceptionV4 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    net = InceptionV4(**kwargs)

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


def inceptionv4(**kwargs):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv4(model_name="inceptionv4", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        inceptionv4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionv4 or weight_count == 42679816)

        x = mx.nd.random.normal(shape=(1, 3, 299, 299), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
