import math

import torch
from torch import nn
from inspect import isfunction

__all__ = ['oth_dla34', 'oth_dla46_c', 'oth_dla46x_c', 'oth_dla60x_c', 'oth_dla60', 'oth_dla60x', 'oth_dla102',
           'oth_dla102x', 'oth_dla102x2', 'oth_dla169']

BatchNorm = nn.BatchNorm2d


def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            assert (activation is not None)
            if isfunction(activation):
                self.activ = activation()
            elif isinstance(activation, str):
                if activation == "relu":
                    self.activ = nn.ReLU(inplace=True)
                elif activation == "relu6":
                    self.activ = nn.ReLU6(inplace=True)
                else:
                    raise NotImplementedError()
            else:
                self.activ = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        activation=activation,
        activate=activate)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        activation=activation,
        activate=activate)


def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=3,
                  bias=False,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    7x7 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        activation=activation,
        activate=activate)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=dilation,
            dilation=dilation)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=dilation,
            dilation=dilation,
            activation=None,
            activate=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 expansion=2):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // expansion
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=dilation,
            dilation=dilation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        x = self.relu(x)
        return x


class BottleneckX(nn.Module):
    cardinality = 32

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        mid_channels = out_channels * cardinality // 32
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=cardinality)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        x = self.relu(x)
        return x


class Root(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 residual):
        super(Root, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=(kernel_size - 1) // 2,
            activation=None,
            activate=False)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):
    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation)
            self.tree2 = block(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                dilation=dilation)
        else:
            self.tree1 = Tree(
                levels=levels - 1,
                block=block,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
            self.tree2 = Tree(
                levels=levels - 1,
                block=block,
                in_channels=out_channels,
                out_channels=out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
        if levels == 1:
            self.root = Root(
                in_channels=root_dim,
                out_channels=out_channels,
                kernel_size=root_kernel_size,
                residual=root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None,
                activate=False)

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 num_classes=1000,
                 block=BasicBlock,
                 residual_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = conv7x7_block(
            in_channels=3,
            out_channels=channels[0])
        self.level0 = self._make_conv_level(
            channels[0],
            channels[0],
            levels[0])
        self.level1 = self._make_conv_level(
            channels[0],
            channels[1],
            levels[1],
            stride=2)

        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            stride=2,
            level_root=False,
            root_residual=residual_root)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            stride=2,
            level_root=True,
            root_residual=residual_root)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            stride=2,
            level_root=True,
            root_residual=residual_root)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            stride=2,
            level_root=True,
            root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = conv1x1(
            in_channels=channels[-1],
            out_channels=num_classes,
            bias=True)

    def _make_conv_level(self,
                         in_channels,
                         out_channels,
                         convs,
                         stride=1,
                         dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=(stride if i == 0 else 1),
                    padding=dilation,
                    dilation=dilation)])
            in_channels = out_channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x


def oth_dla34(pretrained=None, **kwargs):  # DLA-34
    model = DLA(
        [1, 1, 1, 2, 2, 1],
        [16, 32, 64, 128, 256, 512],
        block=BasicBlock,
        **kwargs)
    return model


def oth_dla46_c(pretrained=None, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 2, 1],
        [16, 32, 64, 64, 128, 256],
        block=Bottleneck,
        **kwargs)
    return model


def oth_dla46x_c(pretrained=None, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 2, 1],
        [16, 32, 64, 64, 128, 256],
        block=BottleneckX,
        **kwargs)
    return model


def oth_dla60x_c(pretrained=None, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1],
        [16, 32, 64, 64, 128, 256],
        block=BottleneckX,
        **kwargs)
    return model


def oth_dla60(pretrained=None, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        **kwargs)
    return model


def oth_dla60x(pretrained=None, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 2, 3, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        **kwargs)
    return model


def oth_dla102(pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        residual_root=True,
        **kwargs)
    return model


def oth_dla102x(pretrained=None, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        residual_root=True,
        **kwargs)
    return model


def oth_dla102x2(pretrained=None, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA(
        [1, 1, 1, 3, 4, 1],
        [16, 32, 128, 256, 512, 1024],
        block=BottleneckX,
        residual_root=True,
        **kwargs)
    return model


def oth_dla169(pretrained=None, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA(
        [1, 1, 2, 3, 5, 1],
        [16, 32, 128, 256, 512, 1024],
        block=Bottleneck,
        residual_root=True,
        **kwargs)
    return model


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        oth_dla34,
        oth_dla46_c,
        oth_dla46x_c,
        oth_dla60x_c,
        oth_dla60,
        oth_dla60x,
        oth_dla102,
        oth_dla102x,
        oth_dla102x2,
        oth_dla169,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_dla34 or weight_count == 15783832)
        assert (model != oth_dla46_c or weight_count == 1309848)
        assert (model != oth_dla46x_c or weight_count == 1076888)
        assert (model != oth_dla60x_c or weight_count == 1336728)
        assert (model != oth_dla60 or weight_count == 22334104)
        assert (model != oth_dla60x or weight_count == 17649816)
        assert (model != oth_dla102 or weight_count == 33731736)
        assert (model != oth_dla102x or weight_count == 26772120)
        assert (model != oth_dla102x2 or weight_count == 41745048)
        assert (model != oth_dla169 or weight_count == 53989016)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
