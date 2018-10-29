"""
    InceptionV3, implemented in Gluon.
    Original paper: 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.
"""

__all__ = ['InceptionV3', 'inceptionv3']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent


class InceptConv(HybridBlock):
    """
    InceptionV3 specific convolution block.

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
                epsilon=1e-3,
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


def incept_conv1x1(in_channels,
                   out_channels,
                   bn_use_global_stats):
    """
    1x1 version of the InceptionV3 specific convolution block.

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


class MaxPoolBranch(HybridBlock):
    """
    InceptionV3 specific max pooling branch block.
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
    InceptionV3 specific average pooling branch block.

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
                padding=1)
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
    InceptionV3 specific convolutional 1x1 branch block.

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


class ConvSeqBranch(HybridBlock):
    """
    InceptionV3 specific convolutional sequence branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : tuple of int
        Number of output channels.
    kernel_size : tuple of int or tuple of tuple/list of 2 int
        Convolution window size.
    strides : tuple of int or tuple of tuple/list of 2 int
        Strides of the convolution.
    padding : tuple of int or tuple of tuple/list of 2 int
        Padding value for convolution layer.
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
            self.conv_list = nn.HybridSequential(prefix='')
            for out_channels, kernel_size, strides, padding in zip(
                    out_channels_list, kernel_size_list, strides_list, padding_list):
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
    InceptionV3 specific convolutional sequence branch block with splitting by 3x3.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : tuple of int
        Number of output channels.
    kernel_size : tuple of int or tuple of tuple/list of 2 int
        Convolution window size.
    strides : tuple of int or tuple of tuple/list of 2 int
        Strides of the convolution.
    padding : tuple of int or tuple of tuple/list of 2 int
        Padding value for convolution layer.
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
        super(ConvSeq3x3Branch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv_list = nn.HybridSequential(prefix='')
            for out_channels, kernel_size, strides, padding in zip(
                    out_channels_list, kernel_size_list, strides_list, padding_list):
                self.conv_list.add(InceptConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = out_channels
            self.conv1x3 = InceptConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                strides=1,
                padding=(0, 1),
                bn_use_global_stats=bn_use_global_stats)
            self.conv3x1 = InceptConv(
                in_channels=in_channels,
                out_channels=in_channels,
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


class InceptUnitA(HybridBlock):
    """
    InceptionV3 type A unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    pool_out_channels : int
        Number of output channels in the pool branch.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 pool_out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptUnitA, self).__init__(**kwargs)

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix='')
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=64,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(48, 64),
                kernel_size_list=(1, 5),
                strides_list=(1, 1),
                padding_list=(0, 2),
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
                out_channels=pool_out_channels,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptUnitB(HybridBlock):
    """
    InceptionV3 type B unit.

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
        super(InceptUnitB, self).__init__(**kwargs)

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix='')
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(384,),
                kernel_size_list=(3,),
                strides_list=(2,),
                padding_list=(0,),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(64, 96, 96),
                kernel_size_list=(1, 3, 3),
                strides_list=(1, 1, 2),
                padding_list=(0, 1, 0),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(MaxPoolBranch())

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptUnitC(HybridBlock):
    """
    InceptionV3 type C unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of output channels in the 7x7 branches.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptUnitC, self).__init__(**kwargs)

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix='')
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=192,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(mid_channels, mid_channels, 192),
                kernel_size_list=(1, (1, 7), (7, 1)),
                strides_list=(1, 1, 1),
                padding_list=(0, (0, 3), (3, 0)),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(mid_channels, mid_channels, mid_channels, mid_channels, 192),
                kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
                strides_list=(1, 1, 1, 1, 1),
                padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3)),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(AvgPoolBranch(
                in_channels=in_channels,
                out_channels=192,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptUnitD(HybridBlock):
    """
    InceptionV3 type D unit.

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
        super(InceptUnitD, self).__init__(**kwargs)

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix='')
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(192, 320),
                kernel_size_list=(1, 3),
                strides_list=(1, 2),
                padding_list=(0, 0),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(192, 192, 192, 192),
                kernel_size_list=(1, (1, 7), (7, 1), 3),
                strides_list=(1, 1, 1, 2),
                padding_list=(0, (0, 3), (3, 0), 0),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(MaxPoolBranch())

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptUnitE(HybridBlock):
    """
    InceptionV3 type E unit.

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
        super(InceptUnitE, self).__init__(**kwargs)

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix='')
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=320,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeq3x3Branch(
                in_channels=in_channels,
                out_channels_list=(384,),
                kernel_size_list=(1,),
                strides_list=(1,),
                padding_list=(0,),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeq3x3Branch(
                in_channels=in_channels,
                out_channels_list=(448, 384),
                kernel_size_list=(1, 3),
                strides_list=(1, 1),
                padding_list=(0, 1),
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(AvgPoolBranch(
                in_channels=in_channels,
                out_channels=192,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptInitBlock(HybridBlock):
    """
    InceptionV3 specific initial block.

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
            self.pool1 = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0)
            self.conv4 = InceptConv(
                in_channels=64,
                out_channels=80,
                kernel_size=1,
                strides=1,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)
            self.conv5 = InceptConv(
                in_channels=80,
                out_channels=192,
                kernel_size=3,
                strides=1,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)
            self.pool2 = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        return x


class InceptionV3(HybridBlock):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 **kwargs):
        super(InceptionV3, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(InceptInitBlock(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))

            stage1 = nn.HybridSequential(prefix='stage1_')
            with stage1.name_scope():
                stage1.add(InceptUnitA(
                    in_channels=192,
                    pool_out_channels=32,
                    bn_use_global_stats=bn_use_global_stats))
                stage1.add(InceptUnitA(
                    in_channels=256,
                    pool_out_channels=64,
                    bn_use_global_stats=bn_use_global_stats))
                stage1.add(InceptUnitA(
                    in_channels=288,
                    pool_out_channels=64,
                    bn_use_global_stats=bn_use_global_stats))
            self.features.add(stage1)

            stage2 = nn.HybridSequential(prefix='stage2_')
            with stage2.name_scope():
                stage2.add(InceptUnitB(
                    in_channels=288,
                    bn_use_global_stats=bn_use_global_stats))
                stage2.add(InceptUnitC(
                    in_channels=768,
                    mid_channels=128,
                    bn_use_global_stats=bn_use_global_stats))
                stage2.add(InceptUnitC(
                    in_channels=768,
                    mid_channels=160,
                    bn_use_global_stats=bn_use_global_stats))
                stage2.add(InceptUnitC(
                    in_channels=768,
                    mid_channels=160,
                    bn_use_global_stats=bn_use_global_stats))
                stage2.add(InceptUnitC(
                    in_channels=768,
                    mid_channels=192,
                    bn_use_global_stats=bn_use_global_stats))
            self.features.add(stage2)

            stage3 = nn.HybridSequential(prefix='stage3_')
            with stage3.name_scope():
                stage3.add(InceptUnitD(
                    in_channels=768,
                    bn_use_global_stats=bn_use_global_stats))
                stage3.add(InceptUnitE(
                    in_channels=1280,
                    bn_use_global_stats=bn_use_global_stats))
                stage3.add(InceptUnitE(
                    in_channels=2048,
                    bn_use_global_stats=bn_use_global_stats))
            self.features.add(stage3)

            self.features.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dropout(rate=0.5))
            self.output.add(nn.Dense(
                units=classes,
                in_units=2048))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_inceptionv3(model_name=None,
                    pretrained=False,
                    ctx=cpu(),
                    root=os.path.join('~', '.mxnet', 'models'),
                    **kwargs):
    """
    Create InceptionV3 model with specific parameters.

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

    net = InceptionV3(**kwargs)

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


def inceptionv3(**kwargs):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv3(model_name="inceptionv3", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        inceptionv3,
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
        assert (model != inceptionv3 or weight_count == 23834568)

        x = mx.nd.zeros((1, 3, 299, 299), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
