"""
    SCNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.
"""

__all__ = ['SCNet', 'scnet50', 'scnet101']

import os
import torch
import torch.nn as nn
from .common import conv1x1_block, conv3x3_block, InterpolationBlock
from .resnet import ResInitBlock


class ScDownBlock(nn.Module):
    """
    SCNet specific convolutional downscale block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    pool_size: int or list/tuple of 2 ints, default 2
        Size of the average pooling windows.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_size=2):
        super(ScDownBlock, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=pool_size)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ScConv(nn.Module):
    """
    Self-calibrated convolutional block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    scale_factor : int
        Scale factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 scale_factor):
        super(ScConv, self).__init__()
        self.down = ScDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            pool_size=scale_factor)
        self.up = InterpolationBlock(
            scale_factor=scale_factor,
            mode="nearest",
            align_corners=None)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            activation=None)
        self.conv2 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)

    def forward(self, x):
        w = self.sigmoid(x + self.up(self.down(x), size=x.shape[2:]))
        x = self.conv1(x) * w
        x = self.conv2(x)
        return x


class ScBottleneck(nn.Module):
    """
    SCNet specific bottleneck block for residual path in SCNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    scale_factor : int, default 4
        Scale factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_factor=4,
                 scale_factor=4):
        super(ScBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor // 2

        self.conv1a = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2a = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)

        self.conv1b = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2b = ScConv(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            scale_factor=scale_factor)

        self.conv3 = conv1x1_block(
            in_channels=(2 * mid_channels),
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        y = self.conv1a(x)
        y = self.conv2a(y)

        z = self.conv1b(x)
        z = self.conv2b(z)

        x = torch.cat((y, z), dim=1)

        x = self.conv3(x)
        return x


class ScUnit(nn.Module):
    """
    SCNet unit with residual connection.

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
        super(ScUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ScBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class SCNet(nn.Module):
    """
    SCNet model from 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
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
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(SCNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ScUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_scnet(blocks,
              width_scale=1.0,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create SCNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if blocks == 14:
        layers = [1, 1, 1, 1]
    elif blocks == 26:
        layers = [2, 2, 2, 2]
    elif blocks == 38:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported SCNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    bottleneck_factor = 4
    channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = SCNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def scnet50(**kwargs):
    """
    SCNet-50 model from 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
     http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_scnet(blocks=50, model_name="scnet50b", **kwargs)


def scnet101(**kwargs):
    """
    SCNet-101 model from 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_scnet(blocks=101, model_name="scnet101", **kwargs)


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
        scnet50,
        scnet101,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != scnet50 or weight_count == 25564584)
        assert (model != scnet101 or weight_count == 44565416)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
