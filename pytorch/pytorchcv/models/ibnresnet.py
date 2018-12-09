"""
    IBN-ResNet, implemented in PyTorch.
    Original paper: 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.
"""

__all__ = ['IBNResNet', 'ibnresnet50', 'ibnresnet101', 'ibnresnet152']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block
from .resnet import ResInitBlock


class IBN(nn.Module):
    """
    IBN (Instance-Batch Normalization) block.

    Parameters:
    ----------
    channels : int
        Number of channels.
    """
    def __init__(self,
                 channels):
        super(IBN, self).__init__()
        h1_channels = channels // 2
        h2_channels = channels - h1_channels
        self.half = h1_channels

        self.inst_norm = nn.InstanceNorm2d(h1_channels, affine=True)
        self.batch_norm = nn.BatchNorm2d(h2_channels)

    def forward(self, x_out):
        x_split = torch.split(x_out, split_size_or_sections=self.half, dim=1)
        x1 = self.inst_norm(x_split[0].contiguous())
        x2 = self.batch_norm(x_split[1].contiguous())
        x_out = torch.cat((x1, x2), dim=1)
        return x_out


class IBNNetConv(nn.Module):
    """
    IBN-Net specific convolution block with BN/IBN normalization and ReLU activation.

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
    norm_type : str, default 'bn'
        Name of normalization function to use.
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
                 norm_type="bn",
                 activate=True):
        super(IBNNetConv, self).__init__()
        assert (norm_type in ["bn", "ibn"])
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
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        elif norm_type == "ibn":
            self.norm = IBN(channels=out_channels)
        else:
            raise NotImplementedError()
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activate:
            x = self.activ(x)
        return x


def ibnnet_conv1x1(in_channels,
                   out_channels,
                   stride=1,
                   groups=1,
                   bias=False,
                   norm_type="bn",
                   activate=True):
    """
    1x1 version of the IBN-Net specific convolution block.

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
    norm_type : str, default 'bn'
        Name of normalization function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return IBNNetConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
        norm_type=norm_type,
        activate=activate)


class IBNResBottleneck(nn.Module):
    """
    IBN-ResNet bottleneck block for residual path in IBN-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_ibn : bool
        Whether to use IBN normalization in the first convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 conv1_ibn):
        super(IBNResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = ibnnet_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            norm_type=("ibn" if conv1_ibn else "bn"))
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IBNResUnit(nn.Module):
    """
    IBN-ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_ibn : bool
        Whether to use IBN normalization in the first convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 conv1_ibn):
        super(IBNResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = IBNResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            conv1_ibn=conv1_ibn)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activate=False)
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


class IBNResNet(nn.Module):
    """
    IBN-ResNet model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

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
        super(IBNResNet, self).__init__()
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
                conv1_ibn = (in_channels >= 512)
                stage.add_module("unit{}".format(j + 1), IBNResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv1_ibn=conv1_ibn))
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


def get_ibnresnet(blocks,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join('~', '.torch', 'models'),
                  **kwargs):
    """
    Create IBN-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported AirNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = IBNResNet(
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


def ibnresnet50(**kwargs):
    """
    IBN-ResNet-50 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ibnresnet(blocks=50, model_name="ibnresnet50", **kwargs)


def ibnresnet101(**kwargs):
    """
    IBN-ResNet-101 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ibnresnet(blocks=101, model_name="ibnresnet101", **kwargs)


def ibnresnet152(**kwargs):
    """
    IBN-ResNet-152 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ibnresnet(blocks=152, model_name="ibnresnet152", **kwargs)


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
        ibnresnet50,
        ibnresnet101,
        ibnresnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibnresnet50 or weight_count == 25557032)
        assert (model != ibnresnet101 or weight_count == 44549160)
        assert (model != ibnresnet152 or weight_count == 60192808)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
