"""
    PNASNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Progressive Neural Architecture Search,' https://arxiv.org/abs/1712.00559.
"""

__all__ = ['PNASNet', 'pnasnet5large']


import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1
from .nasnet import nasnet_dual_path_sequential, nasnet_batch_norm, NasConv, NasDwsConv, NasPathBlock, NASNetInitBlock


class PnasMaxPoolBlock(nn.Module):
    """
    PNASNet specific Max pooling layer with extra padding.

    Parameters:
    ----------
    stride : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    def __init__(self,
                 stride=2,
                 extra_padding=False):
        super(PnasMaxPoolBlock, self).__init__()
        self.extra_padding = extra_padding

        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=stride,
            padding=1)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        if self.extra_padding:
            x = self.pad(x)
        x = self.pool(x)
        if self.extra_padding:
            x = x[:, :, 1:, 1:].contiguous()
        return x


def pnas_conv1x1(in_channels,
                 out_channels,
                 stride=1):
    """
    1x1 version of the PNASNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    """
    return NasConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=1)


class DwsBranch(nn.Module):
    """
    PNASNet specific block with depthwise separable convolution layers.

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
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 extra_padding=False,
                 stem=False):
        super(DwsBranch, self).__init__()
        assert (not stem) or (not extra_padding)
        mid_channels = out_channels if stem else in_channels
        padding = kernel_size // 2

        self.conv1 = NasDwsConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            extra_padding=extra_padding)
        self.conv2 = NasDwsConv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def dws_branch_k3(in_channels,
                  out_channels,
                  stride=2,
                  extra_padding=False,
                  stem=False):
    """
    3x3 version of the PNASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        extra_padding=extra_padding,
        stem=stem)


def dws_branch_k5(in_channels,
                  out_channels,
                  stride=2,
                  extra_padding=False,
                  stem=False):
    """
    5x5 version of the PNASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=stride,
        extra_padding=extra_padding,
        stem=stem)


def dws_branch_k7(in_channels,
                  out_channels,
                  stride=2,
                  extra_padding=False):
    """
    7x7 version of the PNASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=stride,
        extra_padding=extra_padding,
        stem=False)


class PnasMaxPathBlock(nn.Module):
    """
    PNASNet specific `max path` auxiliary block.

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
        super(PnasMaxPathBlock, self).__init__()
        self.maxpool = PnasMaxPoolBlock()
        self.conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)
        self.bn = nasnet_batch_norm(channels=out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class PnasBaseUnit(nn.Module):
    """
    PNASNet base unit.
    """
    def __init__(self):
        super(PnasBaseUnit, self).__init__()

    def cell_forward(self, x, x_prev):
        assert (hasattr(self, 'comb0_left'))
        x_left = x_prev
        x_right = x

        x0 = self.comb0_left(x_left) + self.comb0_right(x_left)
        x1 = self.comb1_left(x_right) + self.comb1_right(x_right)
        x2 = self.comb2_left(x_right) + self.comb2_right(x_right)
        x3 = self.comb3_left(x2) + self.comb3_right(x_right)
        x4 = self.comb4_left(x_left) + (self.comb4_right(x_right) if self.comb4_right else x_right)

        x_out = torch.cat((x0, x1, x2, x3, x4), dim=1)
        return x_out


class Stem1Unit(PnasBaseUnit):
    """
    PNASNet Stem1 unit.

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
        mid_channels = out_channels // 5

        self.conv_1x1 = pnas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.comb0_left = dws_branch_k5(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True)
        self.comb0_right = PnasMaxPathBlock(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.comb1_left = dws_branch_k7(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb1_right = PnasMaxPoolBlock()

        self.comb2_left = dws_branch_k5(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.comb2_right = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels)

        self.comb3_left = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=1)
        self.comb3_right = PnasMaxPoolBlock()

        self.comb4_left = dws_branch_k3(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True)
        self.comb4_right = pnas_conv1x1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=2)

    def forward(self, x):
        x_prev = x
        x = self.conv_1x1(x)
        x_out = self.cell_forward(x, x_prev)
        return x_out


class PnasUnit(PnasBaseUnit):
    """
    PNASNet ordinary unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    reduction : bool, default False
        Whether to use reduction.
    extra_padding : bool, default False
        Whether to use extra padding.
    match_prev_layer_dimensions : bool, default False
        Whether to match previous layer dimensions.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 reduction=False,
                 extra_padding=False,
                 match_prev_layer_dimensions=False):
        super(PnasUnit, self).__init__()
        mid_channels = out_channels // 5
        stride = 2 if reduction else 1

        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = NasPathBlock(
                in_channels=prev_in_channels,
                out_channels=mid_channels)
        else:
            self.conv_prev_1x1 = pnas_conv1x1(
                in_channels=prev_in_channels,
                out_channels=mid_channels)

        self.conv_1x1 = pnas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)

        self.comb0_left = dws_branch_k5(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            extra_padding=extra_padding)
        self.comb0_right = PnasMaxPoolBlock(
            stride=stride,
            extra_padding=extra_padding)

        self.comb1_left = dws_branch_k7(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            extra_padding=extra_padding)
        self.comb1_right = PnasMaxPoolBlock(
            stride=stride,
            extra_padding=extra_padding)

        self.comb2_left = dws_branch_k5(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            extra_padding=extra_padding)
        self.comb2_right = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            extra_padding=extra_padding)

        self.comb3_left = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=1)
        self.comb3_right = PnasMaxPoolBlock(
            stride=stride,
            extra_padding=extra_padding)

        self.comb4_left = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            extra_padding=extra_padding)
        if reduction:
            self.comb4_right = pnas_conv1x1(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride)
        else:
            self.comb4_right = None

    def forward(self, x, x_prev):
        # print("x.shape={}, x_prev.shape={}".format(x.shape, x_prev.shape))
        x_prev = self.conv_prev_1x1(x_prev)
        x = self.conv_1x1(x)
        x_out = self.cell_forward(x, x_prev)
        return x_out


class PNASNet(nn.Module):
    """
    PNASNet model from 'Progressive Neural Architecture Search,' https://arxiv.org/abs/1712.00559.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    stem1_blocks_channels : list of 2 int
        Number of output channels for the Stem1 unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (331, 331)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 stem1_blocks_channels,
                 in_channels=3,
                 in_size=(331, 331),
                 num_classes=1000):
        super(PNASNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nasnet_dual_path_sequential(
            return_two=False,
            first_ordinals=2,
            last_ordinals=2)
        self.features.add_module("init_block", NASNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels

        self.features.add_module("stem1_unit", Stem1Unit(
            in_channels=in_channels,
            out_channels=stem1_blocks_channels))
        prev_in_channels = in_channels
        in_channels = stem1_blocks_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nasnet_dual_path_sequential()
            for j, out_channels in enumerate(channels_per_stage):
                reduction = (j == 0)
                extra_padding = (j == 0) and (i not in [0, 2])
                match_prev_layer_dimensions = (j == 1) or ((j == 0) and (i == 0))
                stage.add_module("unit{}".format(j + 1), PnasUnit(
                    in_channels=in_channels,
                    prev_in_channels=prev_in_channels,
                    out_channels=out_channels,
                    reduction=reduction,
                    extra_padding=extra_padding,
                    match_prev_layer_dimensions=match_prev_layer_dimensions))
                prev_in_channels = in_channels
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("activ", nn.ReLU())
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=11,
            stride=1))

        self.output = nn.Sequential()
        self.output.add_module('dropout', nn.Dropout(p=0.5))
        self.output.add_module('fc', nn.Linear(
            in_features=in_channels,
            out_features=num_classes))

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


def get_pnasnet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create PNASNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    repeat = 4
    init_block_channels = 96
    stem_blocks_channels = [270, 540]
    norm_channels = [1080, 2160, 4320]
    channels = [[ci] * repeat for ci in norm_channels]
    stem1_blocks_channels = stem_blocks_channels[0]
    channels[0] = [stem_blocks_channels[1]] + channels[0]

    net = PNASNet(
        channels=channels,
        init_block_channels=init_block_channels,
        stem1_blocks_channels=stem1_blocks_channels,
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


def pnasnet5large(**kwargs):
    """
    PNASNet-5-Large model from 'Progressive Neural Architecture Search,' https://arxiv.org/abs/1712.00559.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_pnasnet(model_name="pnasnet5large", **kwargs)


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
        pnasnet5large,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pnasnet5large or weight_count == 86057668)

        x = torch.randn(1, 3, 331, 331)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
