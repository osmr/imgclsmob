import os
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.init as init


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


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False):
    """
    Convolution 3x3 layer.

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
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
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


class PostActivation(nn.Module):
    """
    Pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PostActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class DualPathSequential(nn.Sequential):
    """
    A sequential container for modules with dual inputs/outputs.
    Modules will be executed in the order they are added.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first modules with single input/output.
    last_ordinals : int, default 0
        Number of the final modules with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a module.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal module.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda module, x1, x2: module(x1, x2)),
                 dual_path_scheme_ordinal=(lambda module, x1, x2: (module(x1), x2))):
        super(DualPathSequential, self).__init__()
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def forward(self, x1, x2=None):
        length = len(self._modules.values())
        for i, module in enumerate(self._modules.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class RiRUnit(nn.Module):
    """
    RiR unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(RiRUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.res_pass_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.trans_pass_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.res_cross_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.trans_cross_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.res_postactiv = PostActivation(in_channels=out_channels)
        self.trans_postactiv = PostActivation(in_channels=out_channels)
        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x_res, x_trans):
        if self.resize_identity:
            x_res_identity = self.identity_conv(x_res)
        else:
            x_res_identity = x_res

        y_res = self.res_cross_conv(x_res)
        y_trans = self.trans_cross_conv(x_trans)
        x_res = self.res_pass_conv(x_res)
        x_trans = self.trans_pass_conv(x_trans)

        x_res = x_res + x_res_identity + y_trans
        x_trans = x_trans + y_res

        x_res = self.res_postactiv(x_res)
        x_trans = self.trans_postactiv(x_trans)

        return x_res, x_trans


class RiRInitBlock(nn.Module):
    """
    RiR initial block.

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
        super(RiRInitBlock, self).__init__()
        self.res_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.trans_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x, _):
        x_res = self.res_conv(x)
        x_trans = self.trans_conv(x)
        return x_res, x_trans


class RiRFinalBlock(nn.Module):
    """
    RiR final block.
    """
    def __init__(self):
        super(RiRFinalBlock, self).__init__()

    def forward(self, x_res, x_trans):
        x = torch.cat((x_res, x_trans), dim=1)
        return x, None


class RiR(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 num_classes=100):
        super(RiR, self).__init__()

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=0,
            last_ordinals=0)
        self.features.add_module("init_block", RiRInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), RiRUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", RiRFinalBlock())
        in_channels = final_block_channels

        self.output = nn.Sequential()
        self.output.add_module('final_conv', conv1x1(
            in_channels=in_channels,
            out_channels=num_classes))
        self.output.add_module('final_pool', nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_rir_cifar(pretrained=False,
                  **kwargs):

    channels = [[48, 48, 48, 48], [96, 96, 96, 96, 96, 96], [192, 192, 192, 192, 192, 192]]
    init_block_channels = 48
    final_block_channels = 384

    net = RiR(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)

    return net


def oth_rir_cifar100(pretrained=False, **kwargs):
    model = get_rir_cifar(pretrained=pretrained, **kwargs)
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
        oth_rir_cifar100,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_rir_cifar100 or weight_count == 9527520)

        x = Variable(torch.randn(1, 3, 32, 32))
        y = net(x)
        assert (tuple(y.size()) == (1, 100))


if __name__ == "__main__":
    _test()
