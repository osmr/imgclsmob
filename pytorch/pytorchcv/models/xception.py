"""
    Xception for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Xception: Deep Learning with Depthwise Separable Convolutions,' https://arxiv.org/abs/1610.02357.
"""

__all__ = ['Xception', 'xception']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block


class DwsConv(nn.Module):
    """
    Depthwise separable convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(DwsConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False)
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DwsConvBlock(nn.Module):
    """
    Depthwise separable convolution block with batchnorm and ReLU pre-activation.

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
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activate):
        super(DwsConvBlock, self).__init__()
        self.activate = activate

        if self.activate:
            self.activ = nn.ReLU(inplace=False)
        self.conv = DwsConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if self.activate:
            x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def dws_conv3x3_block(in_channels,
                      out_channels,
                      activate):
    """
    3x3 version of the depthwise separable convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool
        Whether activate the convolution block.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activate=activate)


class XceptionUnit(nn.Module):
    """
    Xception unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the downsample polling.
    reps : int
        Number of repetitions.
    start_with_relu : bool, default True
        Whether start with ReLU activation.
    grow_first : bool, default True
        Whether start from growing.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 reps,
                 start_with_relu=True,
                 grow_first=True):
        super(XceptionUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)

        self.body = nn.Sequential()
        for i in range(reps):
            if (grow_first and (i == 0)) or ((not grow_first) and (i == reps - 1)):
                in_channels_i = in_channels
                out_channels_i = out_channels
            else:
                if grow_first:
                    in_channels_i = out_channels
                    out_channels_i = out_channels
                else:
                    in_channels_i = in_channels
                    out_channels_i = in_channels
            activate = start_with_relu if (i == 0) else True
            self.body.add_module("block{}".format(i + 1), dws_conv3x3_block(
                in_channels=in_channels_i,
                out_channels=out_channels_i,
                activate=activate))
        if stride != 1:
            self.body.add_module("pool", nn.MaxPool2d(
                kernel_size=3,
                stride=stride,
                padding=1))

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class XceptionInitBlock(nn.Module):
    """
    Xception specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(XceptionInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            stride=2,
            padding=0)
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=64,
            stride=1,
            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class XceptionFinalBlock(nn.Module):
    """
    Xception specific final block.
    """
    def __init__(self):
        super(XceptionFinalBlock, self).__init__()
        self.conv1 = dws_conv3x3_block(
            in_channels=1024,
            out_channels=1536,
            activate=False)
        self.conv2 = dws_conv3x3_block(
            in_channels=1536,
            out_channels=2048,
            activate=True)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(
            kernel_size=10,
            stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class Xception(nn.Module):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 in_channels=3,
                 in_size=(299, 299),
                 num_classes=1000):
        super(Xception, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", XceptionInitBlock(
            in_channels=in_channels))
        in_channels = 64
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), XceptionUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=(2 if (j == 0) else 1),
                    reps=(2 if (j == 0) else 3),
                    start_with_relu=((i != 0) or (j != 0)),
                    grow_first=((i != len(channels) - 1) or (j != len(channels_per_stage) - 1))))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", XceptionFinalBlock())

        self.output = nn.Linear(
            in_features=2048,
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


def get_xception(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".torch", "models"),
                 **kwargs):
    """
    Create Xception model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    channels = [[128], [256], [728] * 9, [1024]]

    net = Xception(
        channels=channels,
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


def xception(**kwargs):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xception(model_name="xception", **kwargs)


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
        xception,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xception or weight_count == 22855952)

        x = torch.randn(1, 3, 299, 299)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
