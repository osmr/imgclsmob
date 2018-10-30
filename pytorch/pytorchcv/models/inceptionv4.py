"""
    InceptionV4, implemented in PyTorch.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionV4', 'inceptionv4']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from common import Concurrent


class InceptConv(nn.Module):
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
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(InceptConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3,
            momentum=0.1)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
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
        kernel_size=1,
        stride=1,
        padding=0)


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
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding)


class MaxPoolBranch(nn.Module):
    """
    InceptionV4 specific max pooling branch block.
    """
    def __init__(self):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x


class AvgPoolBranch(nn.Module):
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
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False)
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Conv1x1Branch(nn.Module):
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
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv3x3Branch(nn.Module):
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
        self.conv = incept_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(nn.Module):
    """
    InceptionV4 specific convolutional sequence branch block.

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

        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_list(x)
        return x


class ConvSeq3x3Branch(nn.Module):
    """
    InceptionV4 specific convolutional sequence branch block with splitting by 3x3.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels_list : tuple of int
        Number of output channels for middle layers.
    kernel_size : tuple of int or tuple of tuple/list of 2 int
        Convolution window size.
    strides : tuple of int or tuple of tuple/list of 2 int
        Strides of the convolution.
    padding : tuple of int or tuple of tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list):
        super(ConvSeq3x3Branch, self).__init__()
        self.conv_list = nn.Sequential()
        for i, (mid_channels, kernel_size, strides, padding) in enumerate(zip(
                mid_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), InceptConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding))
            in_channels = mid_channels
        self.conv1x3 = InceptConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1))
        self.conv3x1 = InceptConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0))

    def forward(self, x):
        x = self.conv_list(x)
        y1 = self.conv1x3(x)
        y2 = self.conv3x1(x)
        x = torch.cat((y1, y2), dim=1)
        return x


class InceptUnit3a(nn.Module):
    """
    InceptionV4 type Mixed-3a unit.
    """
    def __init__(self):
        super(InceptUnit3a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", MaxPoolBranch())
        self.branches.add_module("branch2", Conv3x3Branch(
            in_channels=64,
            out_channels=96))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptUnit4a(nn.Module):
    """
    InceptionV4 type Mixed-4a unit.
    """
    def __init__(self):
        super(InceptUnit4a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 0)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 64, 64, 96),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 1),
            padding_list=(0, (0, 3), (3, 0), 0)))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptUnit5a(nn.Module):
    """
    InceptionV4 type Mixed-5a unit.
    """
    def __init__(self):
        super(InceptUnit5a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv3x3Branch(
            in_channels=192,
            out_channels=192))
        self.branches.add_module("branch2", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptUnitA(nn.Module):
    """
    InceptionV4 type A unit.
    """
    def __init__(self):
        super(InceptUnitA, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=96))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1)))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1)))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=96))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductUnitA(nn.Module):
    """
    InceptionV4 type A-reduction unit.
    """
    def __init__(self):
        super(ReductUnitA, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0)))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptUnitB(nn.Module):
    """
    InceptionV4 type B unit.
    """
    def __init__(self):
        super(InceptUnitB, self).__init__()
        in_channels = 1024

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=384))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0))))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192, 224, 224, 256),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3))))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=128))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductUnitB(nn.Module):
    """
    InceptionV4 type B-reduction unit.
    """
    def __init__(self):
        super(ReductUnitB, self).__init__()
        in_channels = 1024

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 320, 320),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 2),
            padding_list=(0, (0, 3), (3, 0), 0)))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptUnitC(nn.Module):
    """
    InceptionV4 type C unit.
    """
    def __init__(self):
        super(InceptUnitC, self).__init__()
        in_channels = 1536

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=256))
        self.branches.add_module("branch2", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,)))
        self.branches.add_module("branch3", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384, 448, 512),
            kernel_size_list=(1, (3, 1), (1, 3)),
            strides_list=(1, 1, 1),
            padding_list=(0, (1, 0), (0, 1))))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=256))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(nn.Module):
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
        self.conv1 = InceptConv(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0)
        self.conv2 = InceptConv(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0)
        self.conv3 = InceptConv(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.unit1 = InceptUnit3a()
        self.unit2 = InceptUnit4a()
        self.unit3 = InceptUnit5a()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        return x


class InceptionV4(nn.Module):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 in_channels=3,
                 in_size=(299, 299),
                 num_classes=1000):
        super(InceptionV4, self).__init__()
        self.in_size = in_size
        self.classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", InceptInitBlock(
            in_channels=in_channels))

        stage1 = nn.Sequential()
        stage1.add_module("unit1", InceptUnitA())
        stage1.add_module("unit2", InceptUnitA())
        stage1.add_module("unit3", InceptUnitA())
        stage1.add_module("unit4", InceptUnitA())
        self.features.add_module("stage1", stage1)

        stage2 = nn.Sequential()
        stage2.add_module("unit1", ReductUnitA())
        stage2.add_module("unit2", InceptUnitB())
        stage2.add_module("unit3", InceptUnitB())
        stage2.add_module("unit4", InceptUnitB())
        stage2.add_module("unit5", InceptUnitB())
        stage2.add_module("unit6", InceptUnitB())
        stage2.add_module("unit7", InceptUnitB())
        stage2.add_module("unit8", InceptUnitB())
        self.features.add_module("stage2", stage2)

        stage3 = nn.Sequential()
        stage3.add_module("unit1", ReductUnitB())
        stage3.add_module("unit2", InceptUnitC())
        stage3.add_module("unit3", InceptUnitC())
        stage3.add_module("unit4", InceptUnitC())
        self.features.add_module("stage3", stage3)

        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self.output = nn.Sequential()
        self.output.add_module('dropout', nn.Dropout(p=0.5))
        self.output.add_module('fc', nn.Linear(
            in_features=1536,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_inceptionv4(model_name=None,
                    pretrained=False,
                    root=os.path.join('~', '.torch', 'models'),
                    **kwargs):
    """
    Create InceptionV4 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    net = InceptionV4(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def inceptionv4(**kwargs):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv4(model_name="inceptionv4", **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        inceptionv4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != InceptionV4 or weight_count == 42679816)

        x = Variable(torch.randn(1, 3, 299, 299))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
