"""
    DARTS for ImageNet-1K, implemented in PyTorch.
    Original paper: 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.
"""

__all__ = ['DARTS', 'darts']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, Identity
from .nasnet import nasnet_dual_path_sequential


class DwsConv(nn.Module):
    """
    Standard dilated depthwise separable convolution block with.

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
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layers use a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias=False):
        super(DwsConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        self.pw_conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DartsConv(nn.Module):
    """
    DARTS specific convolution block.

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
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activate=True):
        super(DartsConv, self).__init__()
        self.activate = activate

        if self.activate:
            self.activ = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if self.activate:
            x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def darts_conv1x1(in_channels,
                  out_channels,
                  activate=True):
    """
    1x1 version of the DARTS specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return DartsConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        activate=activate)


def darts_conv3x3_s2(in_channels,
                     out_channels,
                     activate=True):
    """
    3x3 version of the DARTS specific convolution block with stride 2.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return DartsConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        activate=activate)


class DartsDwsConv(nn.Module):
    """
    DARTS specific dilated convolution block.

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
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation):
        super(DartsDwsConv, self).__init__()
        self.activ = nn.ReLU(inplace=False)
        self.conv = DwsConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DartsDwsBranch(nn.Module):
    """
    DARTS specific block with depthwise separable convolution layers.

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
        super(DartsDwsBranch, self).__init__()
        mid_channels = in_channels

        self.conv1 = DartsDwsConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1)
        self.conv2 = DartsDwsConv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DartsReduceBranch(nn.Module):
    """
    DARTS specific factorized reduce block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=2):
        super(DartsReduceBranch, self).__init__()
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        self.activ = nn.ReLU(inplace=False)
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.activ(x)
        x1 = self.conv1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        return x


class Stem1Unit(nn.Module):
    """
    DARTS Stem1 unit.

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
        super(Stem1Unit, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = darts_conv3x3_s2(
            in_channels=in_channels,
            out_channels=mid_channels,
            activate=False)
        self.conv2 = darts_conv3x3_s2(
            in_channels=mid_channels,
            out_channels=out_channels,
            activate=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def stem2_unit(in_channels,
               out_channels):
    """
    DARTS Stem2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return darts_conv3x3_s2(
        in_channels=in_channels,
        out_channels=out_channels,
        activate=True)


def darts_maxpool3x3(channels,
                     stride):
    """
    DARTS specific 3x3 Max pooling layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels. Unused parameter.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    assert (channels > 0)
    return nn.MaxPool2d(
        kernel_size=3,
        stride=stride,
        padding=1)


def darts_skip_connection(channels,
                          stride):
    """
    DARTS specific skip connection layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    assert (channels > 0)
    if stride == 1:
        return Identity()
    else:
        assert (stride == 2)
        return DartsReduceBranch(
            in_channels=channels,
            out_channels=channels,
            stride=stride)


def darts_dws_conv3x3(channels,
                      stride):
    """
    3x3 version of DARTS specific dilated convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return DartsDwsConv(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=2,
        dilation=2)


def darts_dws_branch3x3(channels,
                        stride):
    """
    3x3 version of DARTS specific dilated convolution branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return DartsDwsBranch(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1)


# Set of operations in genotype.
GENOTYPE_OPS = {
    'max_pool_3x3': darts_maxpool3x3,
    'skip_connect': darts_skip_connection,
    'dil_conv_3x3': darts_dws_conv3x3,
    'sep_conv_3x3': darts_dws_branch3x3,
}


class DartsMainBlock(nn.Module):
    """
    DARTS main block, described by genotype.

    Parameters:
    ----------
    genotype : list of tuples (str, int)
        List of genotype elements (operations and linked indices).
    channels : int
        Number of input/output channels.
    reduction : bool
        Whether use reduction.
    """
    def __init__(self,
                 genotype,
                 channels,
                 reduction):
        super(DartsMainBlock, self).__init__()
        self.concat = [2, 3, 4, 5]
        op_names, indices = zip(*genotype)
        self.indices = indices
        self.steps = len(op_names) // 2

        self.ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = GENOTYPE_OPS[name](channels, stride)
            self.ops += [op]

    def forward(self, x, x_prev):
        s0 = x_prev
        s1 = x
        states = [s0, s1]
        for i in range(self.steps):
            j1 = 2 * i
            j2 = 2 * i + 1
            op1 = self.ops[j1]
            op2 = self.ops[j2]
            y1 = states[self.indices[j1]]
            y2 = states[self.indices[j2]]
            y1 = op1(y1)
            y2 = op2(y2)
            s = y1 + y2
            states += [s]
        x_out = torch.cat([states[i] for i in self.concat], dim=1)
        return x_out


class DartsUnit(nn.Module):
    """
    DARTS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    genotype : list of tuples (str, int)
        List of genotype elements (operations and linked indices).
    reduction : bool
        Whether use reduction.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 genotype,
                 reduction,
                 prev_reduction):
        super(DartsUnit, self).__init__()
        mid_channels = out_channels // 4

        if prev_reduction:
            self.preprocess_prev = DartsReduceBranch(
                in_channels=prev_in_channels,
                out_channels=mid_channels)
        else:
            self.preprocess_prev = darts_conv1x1(
                in_channels=prev_in_channels,
                out_channels=mid_channels)

        self.preprocess = darts_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.body = DartsMainBlock(
            genotype=genotype,
            channels=mid_channels,
            reduction=reduction)

    def forward(self, x, x_prev):
        x = self.preprocess(x)
        x_prev = self.preprocess_prev(x_prev)
        x_out = self.body(x, x_prev)
        return x_out


class DARTS(nn.Module):
    """
    DARTS model from 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    stem_blocks_channels : int
        Number of output channels for the Stem units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 stem_blocks_channels,
                 normal_genotype,
                 reduce_genotype,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DARTS, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nasnet_dual_path_sequential(
            return_two=False,
            first_ordinals=2,
            last_ordinals=1)
        self.features.add_module("stem1_unit", Stem1Unit(
            in_channels=in_channels,
            out_channels=stem_blocks_channels))
        in_channels = stem_blocks_channels
        self.features.add_module("stem2_unit", stem2_unit(
            in_channels=in_channels,
            out_channels=stem_blocks_channels))
        prev_in_channels = in_channels
        in_channels = stem_blocks_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nasnet_dual_path_sequential()
            for j, out_channels in enumerate(channels_per_stage):
                reduction = (i != 0) and (j == 0)
                prev_reduction = ((i == 0) and (j == 0)) or ((i != 0) and (j == 1))
                genotype = reduce_genotype if reduction else normal_genotype
                stage.add_module("unit{}".format(j + 1), DartsUnit(
                    in_channels=in_channels,
                    prev_in_channels=prev_in_channels,
                    out_channels=out_channels,
                    genotype=genotype,
                    reduction=reduction,
                    prev_reduction=prev_reduction))
                prev_in_channels = in_channels
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

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


def get_darts(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create DARTS model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    stem_blocks_channels = 48
    layers = [4, 5, 5]
    channels_per_layers = [192, 384, 768]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    normal_genotype = [
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)]
    reduce_genotype = [
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)]

    net = DARTS(
        channels=channels,
        stem_blocks_channels=stem_blocks_channels,
        normal_genotype=normal_genotype,
        reduce_genotype=reduce_genotype,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def darts(**kwargs):
    """
    DARTS model from 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_darts(model_name="darts", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        darts,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darts or weight_count == 4718752)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
