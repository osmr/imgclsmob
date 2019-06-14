"""
    BAM-ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.
"""

__all__ = ['BamResNet', 'bam_resnet18', 'bam_resnet34', 'bam_resnet50', 'bam_resnet101', 'bam_resnet152']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, conv1x1_block, conv3x3_block
from .resnet import ResInitBlock, ResUnit


class DenseBlock(nn.Module):
    """
    Standard dense block with Batch normalization and ReLU activation.

    Parameters:
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    """
    def __init__(self,
                 in_features,
                 out_features):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features)
        self.bn = nn.BatchNorm1d(num_features=out_features)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class ChannelGate(nn.Module):
    """
    BAM channel gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_layers : int, default 1
        Number of dense blocks.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 num_layers=1):
        super(ChannelGate, self).__init__()
        mid_channels = channels // reduction_ratio

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.init_fc = DenseBlock(
            in_features=channels,
            out_features=mid_channels)
        self.main_fcs = nn.Sequential()
        for i in range(num_layers - 1):
            self.main_fcs.add_module("fc{}".format(i + 1), DenseBlock(
                in_features=mid_channels,
                out_features=mid_channels))
        self.final_fc = nn.Linear(
            in_features=mid_channels,
            out_features=channels)

    def forward(self, x):
        input = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.init_fc(x)
        x = self.main_fcs(x)
        x = self.final_fc(x)
        x = x.unsqueeze(2).unsqueeze(3).expand_as(input)
        return x


class SpatialGate(nn.Module):
    """
    BAM spatial gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_dil_convs : int, default 2
        Number of dilated convolutions.
    dilation : int, default 4
        Dilation/padding value for corresponding convolutions.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 num_dil_convs=2,
                 dilation=4):
        super(SpatialGate, self).__init__()
        mid_channels = channels // reduction_ratio

        self.init_conv = conv1x1_block(
            in_channels=channels,
            out_channels=mid_channels,
            stride=1,
            bias=True)
        self.dil_convs = nn.Sequential()
        for i in range(num_dil_convs):
            self.dil_convs.add_module("conv{}".format(i + 1), conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=True))
        self.final_conv = conv1x1(
            in_channels=mid_channels,
            out_channels=1,
            stride=1,
            bias=True)

    def forward(self, x):
        input = x
        x = self.init_conv(x)
        x = self.dil_convs(x)
        x = self.final_conv(x)
        x = x.expand_as(input)
        return x


class BamBlock(nn.Module):
    """
    BAM attention block for BAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    """
    def __init__(self,
                 channels):
        super(BamBlock, self).__init__()
        self.ch_att = ChannelGate(channels=channels)
        self.sp_att = SpatialGate(channels=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = 1 + self.sigmoid(self.ch_att(x) * self.sp_att(x))
        x = x * att
        return x


class BamResUnit(nn.Module):
    """
    BAM-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(BamResUnit, self).__init__()
        self.use_bam = (stride != 1)

        if self.use_bam:
            self.bam = BamBlock(channels=in_channels)
        self.res_unit = ResUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck=bottleneck,
            conv1_stride=False)

    def forward(self, x):
        if self.use_bam:
            x = self.bam(x)
        x = self.res_unit(x)
        return x


class BamResNet(nn.Module):
    """
    BAM-ResNet model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
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
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(BamResNet, self).__init__()
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
                stage.add_module("unit{}".format(j + 1), BamResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
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


def get_resnet(blocks,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create BAM-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    use_se : bool
        Whether to use SE block.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported BAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BamResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
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


def bam_resnet18(**kwargs):
    """
    BAM-ResNet-18 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="bam_resnet18", **kwargs)


def bam_resnet34(**kwargs):
    """
    BAM-ResNet-34 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="bam_resnet34", **kwargs)


def bam_resnet50(**kwargs):
    """
    BAM-ResNet-50 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="bam_resnet50", **kwargs)


def bam_resnet101(**kwargs):
    """
    BAM-ResNet-101 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="bam_resnet101", **kwargs)


def bam_resnet152(**kwargs):
    """
    BAM-ResNet-152 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="bam_resnet152", **kwargs)


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
        bam_resnet18,
        bam_resnet34,
        bam_resnet50,
        bam_resnet101,
        bam_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bam_resnet18 or weight_count == 11712503)
        assert (model != bam_resnet34 or weight_count == 21820663)
        assert (model != bam_resnet50 or weight_count == 25915099)
        assert (model != bam_resnet101 or weight_count == 44907227)
        assert (model != bam_resnet152 or weight_count == 60550875)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
