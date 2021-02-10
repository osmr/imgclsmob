"""
    EDANet for image segmentation, implemented in PyTorch.
    Original paper: 'Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1809.06323.
"""

__all__ = ['EDANet', 'edanet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv3x3, conv1x1_block, asym_conv3x3_block, NormActivation, InterpolationBlock


class DownBlock(nn.Module):
    """
    EDANet specific downsample block for the main branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps):
        super(DownBlock, self).__init__()
        self.expand = (in_channels < out_channels)
        mid_channels = out_channels - in_channels if self.expand else out_channels

        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            stride=2)
        if self.expand:
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps)

    def forward(self, x):
        y = self.conv(x)

        if self.expand:
            z = self.pool(x)
            y = torch.cat((y, z), dim=1)

        y = self.norm_activ(y)
        return y


class EDABlock(nn.Module):
    """
    EDANet base block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(EDABlock, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = asym_conv3x3_block(
            channels=channels,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            lw_activation=None)
        self.conv2 = asym_conv3x3_block(
            channels=channels,
            padding=dilation,
            dilation=dilation,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class EDAUnit(nn.Module):
    """
    EDANet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(EDAUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        mid_channels = out_channels - in_channels

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True)
        self.conv2 = EDABlock(
            channels=mid_channels,
            dilation=dilation,
            dropout_rate=dropout_rate,
            bn_eps=bn_eps)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.cat((x, identity), dim=1)
        x = self.activ(x)
        return x


class EDANet(nn.Module):
    """
    EDANet model from 'Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1809.06323.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for the first unit of each stage.
    dilations : list of list of int
        Dilations for blocks.
    growth_rate : int
        Growth rate for numbers of output channels for each non-first unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    num_classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 channels,
                 dilations,
                 growth_rate,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(EDANet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        dropout_rate = 0.02

        self.features = nn.Sequential()
        for i, dilations_per_stage in enumerate(dilations):
            out_channels = channels[i]
            stage = nn.Sequential()
            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), DownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bn_eps=bn_eps))
                else:
                    out_channels += growth_rate
                    stage.add_module("unit{}".format(j + 1), EDAUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dilation=dilation,
                        dropout_rate=dropout_rate,
                        bn_eps=bn_eps))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.head = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = self.up(x)
        return x


def get_edanet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create EDANet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [15, 60, 130, 450]
    dilations = [[0], [0, 1, 1, 1, 2, 2], [0, 2, 2, 4, 4, 8, 8, 16, 16]]
    growth_rate = 40
    bn_eps = 1e-3

    net = EDANet(
        channels=channels,
        dilations=dilations,
        growth_rate=growth_rate,
        bn_eps=bn_eps,
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


def edanet_cityscapes(num_classes=19, **kwargs):
    """
    EDANet model for Cityscapes from 'Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic
    Segmentation,' https://arxiv.org/abs/1809.06323.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_edanet(num_classes=num_classes, model_name="edanet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        edanet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != edanet_cityscapes or weight_count == 689485)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
