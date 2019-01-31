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


class DLABlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(DLABlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DLABottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion=2):
        super(DLABottleneck, self).__init__()
        mid_channels = out_channels // expansion

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DLABottleneckX(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality=32):
        super(DLABottleneckX, self).__init__()
        mid_channels = out_channels * cardinality // 32

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            groups=cardinality)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DLAResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 body_class=DLABlock,
                 return_down=False):
        super(DLAResBlock, self).__init__()
        self.return_down = return_down
        self.downsample = (stride > 1)
        self.project = (in_channels != out_channels)

        self.body = body_class(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if self.downsample:
            self.downsample_pool = nn.MaxPool2d(
                kernel_size=stride,
                stride=stride)
        if self.project:
            self.project_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None,
                activate=False)

    def forward(self, x):
        down = self.downsample_pool(x) if self.downsample else x
        identity = self.project_conv(down) if self.project else down
        if identity is None:
            identity = x
        x = self.body(x)
        x += identity
        x = self.relu(x)
        if self.return_down:
            return x, down
        else:
            return x


class DLARoot(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 residual):
        super(DLARoot, self).__init__()
        self.residual = residual

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x2, x1, extra):
        last_branch = x2
        x = torch.cat((x2, x1) + tuple(extra), dim=1)
        x = self.conv(x)
        if self.residual:
            x += last_branch
        x = self.relu(x)
        return x


class DLATree(nn.Module):
    def __init__(self,
                 levels,
                 in_channels,
                 out_channels,
                 res_body_class,
                 stride,
                 root_residual,
                 root_dim=0,
                 first_tree=False,
                 input_level=True,
                 return_down=False):
        super(DLATree, self).__init__()
        self.return_down = return_down
        self.add_down = (input_level and not first_tree)
        self.root_level = (levels == 1)

        if root_dim == 0:
            root_dim = 2 * out_channels
        if self.add_down:
            root_dim += in_channels

        if self.root_level:
            self.tree1 = DLAResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                body_class=res_body_class,
                return_down=True)
            self.tree2 = DLAResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                body_class=res_body_class,
                return_down=False)
        else:
            self.tree1 = DLATree(
                levels=levels - 1,
                in_channels=in_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                stride=stride,
                root_residual=root_residual,
                root_dim=0,
                input_level=False,
                return_down=True)
            self.tree2 = DLATree(
                levels=levels - 1,
                in_channels=out_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                stride=1,
                root_residual=root_residual,
                root_dim=root_dim + out_channels,
                input_level=False,
                return_down=False)
        if self.root_level:
            self.root = DLARoot(
                in_channels=root_dim,
                out_channels=out_channels,
                residual=root_residual)

    def forward(self, x, extra=None):
        extra = [] if extra is None else extra
        x1, down = self.tree1(x)
        if self.add_down:
            extra.append(down)
        if self.root_level:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, extra)
        else:
            extra.append(x1)
            x = self.tree2(x1, extra)
        if self.return_down:
            return x, down
        else:
            return x


class DLAInitBlock(nn.Module):
    """
    DLA specific initial block.

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
        super(DLAInitBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DLA(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 init_block_channels,
                 res_body_class,
                 residual_root=False,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DLA, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", DLAInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels

        for i in range(len(levels)):
            levels_i = levels[i]
            out_channels = channels[i]
            first_tree = (i == 0)
            self.features.add_module("stage{}".format(i + 1), DLATree(
                levels=levels_i,
                in_channels=in_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                stride=2,
                root_residual=residual_root,
                first_tree=first_tree))
            in_channels = out_channels

        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def oth_dla34(pretrained=None, **kwargs):  # DLA-34
    model = DLA(
        levels=[1, 2, 2, 1],
        channels=[64, 128, 256, 512],
        init_block_channels=32,
        res_body_class=DLABlock,
        **kwargs)
    return model


def oth_dla46_c(pretrained=None, **kwargs):  # DLA-46-C

    model = DLA(
        levels=[1, 2, 2, 1],
        channels=[64, 64, 128, 256],
        init_block_channels=32,
        res_body_class=DLABottleneck,
        **kwargs)
    return model


def oth_dla46x_c(pretrained=None, **kwargs):  # DLA-X-46-C
    model = DLA(
        levels=[1, 2, 2, 1],
        channels=[64, 64, 128, 256],
        init_block_channels=32,
        res_body_class=DLABottleneckX,
        **kwargs)
    return model


def oth_dla60x_c(pretrained=None, **kwargs):  # DLA-X-60-C
    model = DLA(
        levels=[1, 2, 3, 1],
        channels=[64, 64, 128, 256],
        init_block_channels=32,
        res_body_class=DLABottleneckX,
        **kwargs)
    return model


def oth_dla60(pretrained=None, **kwargs):  # DLA-60
    model = DLA(
        levels=[1, 2, 3, 1],
        channels=[128, 256, 512, 1024],
        init_block_channels=32,
        res_body_class=DLABottleneck,
        **kwargs)
    return model


def oth_dla60x(pretrained=None, **kwargs):  # DLA-X-60
    model = DLA(
        levels=[1, 2, 3, 1],
        channels=[128, 256, 512, 1024],
        init_block_channels=32,
        res_body_class=DLABottleneckX,
        **kwargs)
    return model


def oth_dla102(pretrained=None, **kwargs):  # DLA-102
    model = DLA(
        levels=[1, 3, 4, 1],
        channels=[128, 256, 512, 1024],
        init_block_channels=32,
        res_body_class=DLABottleneck,
        residual_root=True,
        **kwargs)
    return model


def oth_dla102x(pretrained=None, **kwargs):  # DLA-X-102
    model = DLA(
        levels=[1, 3, 4, 1],
        channels=[128, 256, 512, 1024],
        init_block_channels=32,
        res_body_class=DLABottleneckX,
        residual_root=True,
        **kwargs)
    return model


def oth_dla102x2(pretrained=None, **kwargs):  # DLA-X-102 64

    class DLABottleneckX64(DLABottleneckX):
        def __init__(self, in_channels, out_channels, stride):
            super(DLABottleneckX64, self).__init__(in_channels, out_channels, stride, cardinality=64)

    model = DLA(
        levels=[1, 3, 4, 1],
        channels=[128, 256, 512, 1024],
        init_block_channels=32,
        res_body_class=DLABottleneckX64,
        residual_root=True,
        **kwargs)
    return model


def oth_dla169(pretrained=None, **kwargs):  # DLA-169
    model = DLA(
        levels=[2, 3, 5, 1],
        channels=[128, 256, 512, 1024],
        init_block_channels=32,
        res_body_class=DLABottleneck,
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
        assert (model != oth_dla34 or weight_count == 15742104)  # 15783832
        assert (model != oth_dla46_c or weight_count == 1301400)  # 1309848
        assert (model != oth_dla46x_c or weight_count == 1068440)  # 1076888
        assert (model != oth_dla60x_c or weight_count == 1319832)  # 1336728
        assert (model != oth_dla60 or weight_count == 22036632)  # 22334104
        assert (model != oth_dla60x or weight_count == 17352344)  # 17649816
        assert (model != oth_dla102 or weight_count == 33268888)  # 33731736
        assert (model != oth_dla102x or weight_count == 26309272)  # 26772120
        assert (model != oth_dla102x2 or weight_count == 41282200)  # 41745048
        assert (model != oth_dla169 or weight_count == 53389720)  # 53989016

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
