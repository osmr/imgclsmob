"""
    MixNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.
"""

__all__ = ['MixNet', 'mixnet_s', 'mixnet_m', 'mixnet_l']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import round_channels, get_activation_layer, conv1x1_block, conv3x3_block, dwconv3x3_block, SEBlock


class MixConv(nn.Module):
    """
    Mixed convolution layer from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    axis : int, default 1
        The axis on which to concatenate the outputs.
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
                 axis=1):
        super(MixConv, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        padding = padding if isinstance(padding, list) else [padding]
        kernel_count = len(kernel_size)
        self.splitted_in_channels = self.split_channels(in_channels, kernel_count)
        splitted_out_channels = self.split_channels(out_channels, kernel_count)
        for i, kernel_size_i in enumerate(kernel_size):
            in_channels_i = self.splitted_in_channels[i]
            out_channels_i = splitted_out_channels[i]
            padding_i = padding[i]
            self.add_module(
                name=str(i),
                module=nn.Conv2d(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i,
                    kernel_size=kernel_size_i,
                    stride=stride,
                    padding=padding_i,
                    dilation=dilation,
                    groups=(out_channels_i if out_channels == groups else groups),
                    bias=bias))
        self.axis = axis

    def forward(self, x):
        xx = torch.split(x, self.splitted_in_channels, dim=self.axis)
        out = [conv_i(x_i) for x_i, conv_i in zip(xx, self._modules.values())]
        x = torch.cat(tuple(out), dim=self.axis)
        return x

    @staticmethod
    def split_channels(channels, kernel_count):
        splitted_channels = [channels // kernel_count] * kernel_count
        splitted_channels[0] += channels - sum(splitted_channels)
        return splitted_channels


class MixConvBlock(nn.Module):
    """
    Mixed convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
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
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(MixConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = MixConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def mixconv1x1_block(in_channels,
                     out_channels,
                     kernel_count,
                     stride=1,
                     groups=1,
                     bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation=(lambda: nn.ReLU(inplace=True))):
    """
    1x1 version of the mixed convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_count : int
        Kernel count.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str, or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return MixConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=([1] * kernel_count),
        stride=stride,
        padding=([0] * kernel_count),
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class MixUnit(nn.Module):
    """
    MixNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    exp_channels : int
        Number of middle (expanded) channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    exp_kernel_count : int
        Expansion convolution kernel count for each unit.
    conv1_kernel_count : int
        Conv1 kernel count for each unit.
    conv2_kernel_count : int
        Conv2 kernel count for each unit.
    exp_factor : int
        Expansion factor for each unit.
    se_factor : int
        SE reduction factor for each unit.
    activation : str
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 exp_kernel_count,
                 conv1_kernel_count,
                 conv2_kernel_count,
                 exp_factor,
                 se_factor,
                 activation):
        super(MixUnit, self).__init__()
        assert (exp_factor >= 1)
        assert (se_factor >= 0)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = exp_factor * in_channels
        self.use_exp_conv = exp_factor > 1

        if self.use_exp_conv:
            if exp_kernel_count == 1:
                self.exp_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    activation=activation)
            else:
                self.exp_conv = mixconv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_count=exp_kernel_count,
                    activation=activation)
        if conv1_kernel_count == 1:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        else:
            self.conv1 = MixConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=[3 + 2 * i for i in range(conv1_kernel_count)],
                stride=stride,
                padding=[1 + i for i in range(conv1_kernel_count)],
                groups=mid_channels,
                activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                approx_sigmoid=False,
                round_mid=False,
                activation=activation)
        if conv2_kernel_count == 1:
            self.conv2 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None)
        else:
            self.conv2 = mixconv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_count=conv2_kernel_count,
                activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MixInitBlock(nn.Module):
    """
    MixNet specific initial block.

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
        super(MixInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)
        self.conv2 = MixUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            exp_kernel_count=1,
            conv1_kernel_count=1,
            conv2_kernel_count=1,
            exp_factor=1,
            se_factor=0,
            activation="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MixNet(nn.Module):
    """
    MixNet model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    exp_kernel_counts : list of list of int
        Expansion convolution kernel count for each unit.
    conv1_kernel_counts : list of list of int
        Conv1 kernel count for each unit.
    conv2_kernel_counts : list of list of int
        Conv2 kernel count for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    se_factors : list of list of int
        SE reduction factor for each unit.
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
                 final_block_channels,
                 exp_kernel_counts,
                 conv1_kernel_counts,
                 conv2_kernel_counts,
                 exp_factors,
                 se_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MixNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", MixInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if ((j == 0) and (i != 3)) or ((j == len(channels_per_stage) // 2) and (i == 3)) else 1
                exp_kernel_count = exp_kernel_counts[i][j]
                conv1_kernel_count = conv1_kernel_counts[i][j]
                conv2_kernel_count = conv2_kernel_counts[i][j]
                exp_factor = exp_factors[i][j]
                se_factor = se_factors[i][j]
                activation = "relu" if i == 0 else "swish"
                stage.add_module("unit{}".format(j + 1), MixUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    exp_kernel_count=exp_kernel_count,
                    conv1_kernel_count=conv1_kernel_count,
                    conv2_kernel_count=conv2_kernel_count,
                    exp_factor=exp_factor,
                    se_factor=se_factor,
                    activation=activation))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels))
        in_channels = final_block_channels
        self.features.add_module('final_pool', nn.AvgPool2d(
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


def get_mixnet(version,
               width_scale,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create MixNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('s' or 'm').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if version == "s":
        init_block_channels = 16
        channels = [[24, 24], [40, 40, 40, 40], [80, 80, 80], [120, 120, 120, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 1, 1], [2, 2, 2, 1, 1, 1]]
        conv1_kernel_counts = [[1, 1], [3, 2, 2, 2], [3, 2, 2], [3, 4, 4, 5, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [2, 2, 2], [2, 2, 2, 1, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6], [6, 3, 3, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4], [2, 2, 2, 2, 2, 2]]
    elif version == "m":
        init_block_channels = 24
        channels = [[32, 32], [40, 40, 40, 40], [80, 80, 80, 80], [120, 120, 120, 120, 200, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 2, 1, 1, 1, 1]]
        conv1_kernel_counts = [[3, 1], [4, 2, 2, 2], [3, 4, 4, 4], [1, 4, 4, 4, 4, 4, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 2, 1, 2, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6, 6], [6, 3, 3, 3, 6, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4, 4], [2, 2, 2, 2, 2, 2, 2, 2]]
    else:
        raise ValueError("Unsupported MixNet version {}".format(version))

    final_block_channels = 1536

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale)

    net = MixNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        exp_kernel_counts=exp_kernel_counts,
        conv1_kernel_counts=conv1_kernel_counts,
        conv2_kernel_counts=conv2_kernel_counts,
        exp_factors=exp_factors,
        se_factors=se_factors,
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


def mixnet_s(**kwargs):
    """
    MixNet-S model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="s", width_scale=1.0, model_name="mixnet_s", **kwargs)


def mixnet_m(**kwargs):
    """
    MixNet-M model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="m", width_scale=1.0, model_name="mixnet_m", **kwargs)


def mixnet_l(**kwargs):
    """
    MixNet-L model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="m", width_scale=1.3, model_name="mixnet_l", **kwargs)


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
        mixnet_s,
        mixnet_m,
        mixnet_l,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mixnet_s or weight_count == 4134606)
        assert (model != mixnet_m or weight_count == 5014382)
        assert (model != mixnet_l or weight_count == 7329252)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
