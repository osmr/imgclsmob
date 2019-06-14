"""
    Oct-ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
    Convolution,' https://arxiv.org/abs/1904.05049.
"""

__all__ = ['OctResNet', 'octresnet10_ad2', 'octresnet50b_ad2', 'OctResUnit']

import os
from inspect import isfunction
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .common import DualPathSequential
from .resnet import ResInitBlock


class OctConv(nn.Conv2d):
    """
    Octave convolution layer.

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
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    oct_value : int, default 2
        Octave value.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 oct_alpha=0.0,
                 oct_mode="std",
                 oct_value=2):
        if isinstance(stride, int):
            stride = (stride, stride)
        self.downsample = (stride[0] > 1) or (stride[1] > 1)
        assert (stride[0] in [1, oct_value]) and (stride[1] in [1, oct_value])
        stride = (1, 1)
        if oct_mode == "first":
            in_alpha = 0.0
            out_alpha = oct_alpha
        elif oct_mode == "norm":
            in_alpha = oct_alpha
            out_alpha = oct_alpha
        elif oct_mode == "last":
            in_alpha = oct_alpha
            out_alpha = 0.0
        elif oct_mode == "std":
            in_alpha = 0.0
            out_alpha = 0.0
        else:
            raise ValueError("Unsupported octave convolution mode: {}".format(oct_mode))
        self.h_in_channels = int(in_channels * (1.0 - in_alpha))
        self.h_out_channels = int(out_channels * (1.0 - out_alpha))
        self.l_out_channels = out_channels - self.h_out_channels
        self.oct_alpha = oct_alpha
        self.oct_mode = oct_mode
        self.oct_value = oct_value
        super(OctConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.conv_kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups}

    def forward(self, hx, lx=None):
        if self.oct_mode == "std":
            return F.conv2d(
                input=hx,
                weight=self.weight,
                bias=self.bias,
                **self.conv_kwargs), None

        if self.downsample:
            hx = F.avg_pool2d(
                input=hx,
                kernel_size=(self.oct_value, self.oct_value),
                stride=(self.oct_value, self.oct_value))

        hhy = F.conv2d(
            input=hx,
            weight=self.weight[0:self.h_out_channels, 0:self.h_in_channels, :, :],
            bias=self.bias[0:self.h_out_channels] if self.bias is not None else None,
            **self.conv_kwargs)

        if self.oct_mode != "first":
            hlx = F.conv2d(
                input=lx,
                weight=self.weight[0:self.h_out_channels, self.h_in_channels:, :, :],
                bias=self.bias[0:self.h_out_channels] if self.bias is not None else None,
                **self.conv_kwargs)

        if self.oct_mode == "last":
            hy = hhy + hlx
            ly = None
            return hy, ly

        lhx = F.avg_pool2d(
            input=hx,
            kernel_size=(self.oct_value, self.oct_value),
            stride=(self.oct_value, self.oct_value))
        lhy = F.conv2d(
            input=lhx,
            weight=self.weight[self.h_out_channels:, 0:self.h_in_channels, :, :],
            bias=self.bias[self.h_out_channels:] if self.bias is not None else None,
            **self.conv_kwargs)

        if self.oct_mode == "first":
            hy = hhy
            ly = lhy
            return hy, ly

        if self.downsample:
            hly = hlx
            llx = F.avg_pool2d(
                input=lx,
                kernel_size=(self.oct_value, self.oct_value),
                stride=(self.oct_value, self.oct_value))
        else:
            hly = F.interpolate(
                input=hlx,
                scale_factor=self.oct_value,
                mode="nearest")
            llx = lx
        lly = F.conv2d(
            input=llx,
            weight=self.weight[self.h_out_channels:, self.h_in_channels:, :, :],
            bias=self.bias[self.h_out_channels:] if self.bias is not None else None,
            **self.conv_kwargs)

        hy = hhy + hly
        ly = lhy + lly
        return hy, ly


class OctConvBlock(nn.Module):
    """
    Octave convolution block with Batch normalization and ReLU/ReLU6 activation.

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
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 oct_alpha=0.0,
                 oct_mode="std",
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 activate=True):
        super(OctConvBlock, self).__init__()
        self.activate = activate
        self.last = (oct_mode == "last") or (oct_mode == "std")
        out_alpha = 0.0 if self.last else oct_alpha
        h_out_channels = int(out_channels * (1.0 - out_alpha))
        l_out_channels = out_channels - h_out_channels

        self.conv = OctConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            oct_alpha=oct_alpha,
            oct_mode=oct_mode)
        self.h_bn = nn.BatchNorm2d(
            num_features=h_out_channels,
            eps=bn_eps)
        if not self.last:
            self.l_bn = nn.BatchNorm2d(
                num_features=l_out_channels,
                eps=bn_eps)
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

    def forward(self, hx, lx=None):
        hx, lx = self.conv(hx, lx)
        hx = self.h_bn(hx)
        if self.activate:
            hx = self.activ(hx)
        if not self.last:
            lx = self.l_bn(lx)
            if self.activate:
                lx = self.activ(lx)
        return hx, lx


def oct_conv1x1_block(in_channels,
                      out_channels,
                      stride=1,
                      groups=1,
                      bias=False,
                      oct_alpha=0.0,
                      oct_mode="std",
                      bn_eps=1e-5,
                      activation=(lambda: nn.ReLU(inplace=True)),
                      activate=True):
    """
    1x1 version of the octave convolution block.

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
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return OctConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
        oct_alpha=oct_alpha,
        oct_mode=oct_mode,
        bn_eps=bn_eps,
        activation=activation,
        activate=activate)


def oct_conv3x3_block(in_channels,
                      out_channels,
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      bias=False,
                      oct_alpha=0.0,
                      oct_mode="std",
                      bn_eps=1e-5,
                      activation=(lambda: nn.ReLU(inplace=True)),
                      activate=True):
    """
    3x3 version of the octave convolution block.

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
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return OctConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        oct_alpha=oct_alpha,
        oct_mode=oct_mode,
        bn_eps=bn_eps,
        activation=activation,
        activate=activate)


class OctResBlock(nn.Module):
    """
    Simple Oct-ResNet block for residual path in Oct-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 oct_alpha=0.0,
                 oct_mode="std"):
        super(OctResBlock, self).__init__()
        self.conv1 = oct_conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            oct_alpha=oct_alpha,
            oct_mode=oct_mode)
        self.conv2 = oct_conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            oct_alpha=oct_alpha,
            oct_mode=("std" if oct_mode == "last" else (oct_mode if oct_mode != "first" else "norm")),
            activation=None,
            activate=False)

    def forward(self, hx, lx=None):
        hx, lx = self.conv1(hx, lx)
        hx, lx = self.conv2(hx, lx)
        return hx, lx


class OctResBottleneck(nn.Module):
    """
    Oct-ResNet bottleneck block for residual path in Oct-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 oct_alpha=0.0,
                 oct_mode="std",
                 conv1_stride=False,
                 bottleneck_factor=4):
        super(OctResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = oct_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1),
            oct_alpha=oct_alpha,
            oct_mode=(oct_mode if oct_mode != "last" else "norm"))
        self.conv2 = oct_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride),
            padding=padding,
            dilation=dilation,
            oct_alpha=oct_alpha,
            oct_mode=(oct_mode if oct_mode != "first" else "norm"))
        self.conv3 = oct_conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            oct_alpha=oct_alpha,
            oct_mode=("std" if oct_mode == "last" else (oct_mode if oct_mode != "first" else "norm")),
            activation=None,
            activate=False)

    def forward(self, hx, lx=None):
        hx, lx = self.conv1(hx, lx)
        hx, lx = self.conv2(hx, lx)
        hx, lx = self.conv3(hx, lx)
        return hx, lx


class OctResUnit(nn.Module):
    """
    Oct-ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 oct_alpha=0.0,
                 oct_mode="std",
                 bottleneck=True,
                 conv1_stride=False):
        super(OctResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1) or \
                               ((oct_mode == "first") and (oct_alpha != 0.0))

        if bottleneck:
            self.body = OctResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                oct_alpha=oct_alpha,
                oct_mode=oct_mode,
                conv1_stride=conv1_stride)
        else:
            self.body = OctResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                oct_alpha=oct_alpha,
                oct_mode=oct_mode)
        if self.resize_identity:
            self.identity_conv = oct_conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                oct_alpha=oct_alpha,
                oct_mode=oct_mode,
                activation=None,
                activate=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, hx, lx=None):
        if self.resize_identity:
            h_identity, l_identity = self.identity_conv(hx, lx)
        else:
            h_identity, l_identity = hx, lx
        hx, lx = self.body(hx, lx)
        hx = hx + h_identity
        hx = self.activ(hx)
        if lx is not None:
            lx = lx + l_identity
            lx = self.activ(lx)
        return hx, lx


class OctResNet(nn.Module):
    """
    Oct-ResNet model from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
    Convolution,' https://arxiv.org/abs/1904.05049.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    oct_alpha : float, default 0.5
        Octave alpha coefficient.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 oct_alpha=0.5,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(OctResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=1)
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                if (i == 0) and (j == 0):
                    oct_mode = "first"
                elif (i == len(channels) - 1) and (j == 0):
                    oct_mode = "last"
                elif (i == len(channels) - 1) and (j != 0):
                    oct_mode = "std"
                else:
                    oct_mode = "norm"
                stage.add_module("unit{}".format(j + 1), OctResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    oct_alpha=oct_alpha,
                    oct_mode=oct_mode,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
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


def get_octresnet(blocks,
                  bottleneck=None,
                  conv1_stride=True,
                  oct_alpha=0.5,
                  width_scale=1.0,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create Oct-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    oct_alpha : float, default 0.5
        Octave alpha coefficient.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    elif blocks == 269:
        layers = [3, 30, 48, 8]
    else:
        raise ValueError("Unsupported Oct-ResNet with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = OctResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        oct_alpha=oct_alpha,
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


def octresnet10_ad2(**kwargs):
    """
    Oct-ResNet-10 (alpha=1/2) model from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks
    with Octave Convolution,' https://arxiv.org/abs/1904.05049.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_octresnet(blocks=10, oct_alpha=0.5, model_name="octresnet10_ad2", **kwargs)


def octresnet50b_ad2(**kwargs):
    """
    Oct-ResNet-50b (alpha=1/2) model from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks
    with Octave Convolution,' https://arxiv.org/abs/1904.05049.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_octresnet(blocks=50, conv1_stride=False, oct_alpha=0.5, model_name="octresnet50b_ad2", **kwargs)


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
        octresnet10_ad2,
        octresnet50b_ad2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != octresnet10_ad2 or weight_count == 5423016)
        assert (model != octresnet50b_ad2 or weight_count == 25557032)

        x = torch.randn(14, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, 1000))


if __name__ == "__main__":
    _test()
