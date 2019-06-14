"""
    DPN for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.
"""

__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn98', 'dpn107', 'dpn131']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, DualPathSequential


class GlobalAvgMaxPool2D(nn.Module):
    """
    Global average+max pooling operation for spatial data.

    Parameters:
    ----------
    output_size : int, default 1
        The target output size.
    """
    def __init__(self,
                 output_size=1):
        super(GlobalAvgMaxPool2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=output_size)

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = 0.5 * (x_avg + x_max)
        return x


def dpn_batch_norm(channels):
    """
    DPN specific Batch normalization layer.

    Parameters:
    ----------
    channels : int
        Number of channels in input data.
    """
    return nn.BatchNorm2d(
        num_features=channels,
        eps=0.001)


class PreActivation(nn.Module):
    """
    DPN specific block, which performs the preactivation like in RreResNet.

    Parameters:
    ----------
    channels : int
        Number of channels.
    """
    def __init__(self,
                 channels):
        super(PreActivation, self).__init__()
        self.bn = dpn_batch_norm(channels=channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class DPNConv(nn.Module):
    """
    DPN specific convolution block.

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
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups):
        super(DPNConv, self).__init__()
        self.bn = dpn_batch_norm(channels=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def dpn_conv1x1(in_channels,
                out_channels,
                stride=1):
    """
    1x1 version of the DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    """
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=1)


def dpn_conv3x3(in_channels,
                out_channels,
                stride,
                groups):
    """
    3x3 version of the DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int
        Number of groups.
    """
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups)


class DPNUnit(nn.Module):
    """
    DPN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of intermediate channels.
    bw : int
        Number of residual channels.
    inc : int
        Incrementing step for channels.
    groups : int
        Number of groups in the units.
    has_proj : bool
        Whether to use projection.
    key_stride : int
        Key strides of the convolutions.
    b_case : bool, default False
        Whether to use B-case model.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 bw,
                 inc,
                 groups,
                 has_proj,
                 key_stride,
                 b_case=False):
        super(DPNUnit, self).__init__()
        self.bw = bw
        self.has_proj = has_proj
        self.b_case = b_case

        if self.has_proj:
            self.conv_proj = dpn_conv1x1(
                in_channels=in_channels,
                out_channels=bw + 2 * inc,
                stride=key_stride)

        self.conv1 = dpn_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = dpn_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=key_stride,
            groups=groups)

        if b_case:
            self.preactiv = PreActivation(channels=mid_channels)
            self.conv3a = conv1x1(
                in_channels=mid_channels,
                out_channels=bw)
            self.conv3b = conv1x1(
                in_channels=mid_channels,
                out_channels=inc)
        else:
            self.conv3 = dpn_conv1x1(
                in_channels=mid_channels,
                out_channels=bw + inc)

    def forward(self, x1, x2=None):
        x_in = torch.cat((x1, x2), dim=1) if x2 is not None else x1
        if self.has_proj:
            x_s = self.conv_proj(x_in)
            x_s1 = x_s[:, :self.bw, :, :]
            x_s2 = x_s[:, self.bw:, :, :]
        else:
            assert (x2 is not None)
            x_s1 = x1
            x_s2 = x2
        x_in = self.conv1(x_in)
        x_in = self.conv2(x_in)
        if self.b_case:
            x_in = self.preactiv(x_in)
            y1 = self.conv3a(x_in)
            y2 = self.conv3b(x_in)
        else:
            x_in = self.conv3(x_in)
            y1 = x_in[:, :self.bw, :, :]
            y2 = x_in[:, self.bw:, :, :]
        residual = x_s1 + y1
        dense = torch.cat((x_s2, y2), dim=1)
        return residual, dense


class DPNInitBlock(nn.Module):
    """
    DPN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding):
        super(DPNInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False)
        self.bn = dpn_batch_norm(channels=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class DPNFinalBlock(nn.Module):
    """
    DPN final block, which performs the preactivation with cutting.

    Parameters:
    ----------
    channels : int
        Number of channels.
    """
    def __init__(self,
                 channels):
        super(DPNFinalBlock, self).__init__()
        self.activ = PreActivation(channels=channels)

    def forward(self, x1, x2):
        assert (x2 is not None)
        x = torch.cat((x1, x2), dim=1)
        x = self.activ(x)
        return x, None


class DPN(nn.Module):
    """
    DPN model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    init_block_kernel_size : int or tuple/list of 2 int
        Convolution window size for the initial unit.
    init_block_padding : int or tuple/list of 2 int
        Padding value for convolution layer in the initial unit.
    rs : list f int
        Number of intermediate channels for each unit.
    bws : list f int
        Number of residual channels for each unit.
    incs : list f int
        Incrementing step for channels for each unit.
    groups : int
        Number of groups in the units.
    b_case : bool
        Whether to use B-case model.
    for_training : bool
        Whether to use model for training.
    test_time_pool : bool
        Whether to use the avg-max pooling in the inference mode.
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
                 init_block_kernel_size,
                 init_block_padding,
                 rs,
                 bws,
                 incs,
                 groups,
                 b_case,
                 for_training,
                 test_time_pool,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DPN, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=0)
        self.features.add_module("init_block", DPNInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=init_block_kernel_size,
            padding=init_block_padding))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            r = rs[i]
            bw = bws[i]
            inc = incs[i]
            for j, out_channels in enumerate(channels_per_stage):
                has_proj = (j == 0)
                key_stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), DPNUnit(
                    in_channels=in_channels,
                    mid_channels=r,
                    bw=bw,
                    inc=inc,
                    groups=groups,
                    has_proj=has_proj,
                    key_stride=key_stride,
                    b_case=b_case))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', DPNFinalBlock(channels=in_channels))

        self.output = nn.Sequential()
        if for_training or not test_time_pool:
            self.output.add_module('final_pool', nn.AdaptiveAvgPool2d(output_size=1))
            self.output.add_module('classifier', conv1x1(
                in_channels=in_channels,
                out_channels=num_classes,
                bias=True))
        else:
            self.output.add_module('avg_pool', nn.AvgPool2d(
                kernel_size=7,
                stride=1))
            self.output.add_module('classifier', conv1x1(
                in_channels=in_channels,
                out_channels=num_classes,
                bias=True))
            self.output.add_module('avgmax_pool', GlobalAvgMaxPool2D())

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


def get_dpn(num_layers,
            b_case=False,
            for_training=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".torch", "models"),
            **kwargs):
    """
    Create DPN model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    b_case : bool, default False
        Whether to use B-case model.
    for_training : bool
        Whether to use model for training.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if num_layers == 68:
        init_block_channels = 10
        init_block_kernel_size = 3
        init_block_padding = 1
        bw_factor = 1
        k_r = 128
        groups = 32
        k_sec = (3, 4, 12, 3)
        incs = (16, 32, 32, 64)
        test_time_pool = True
    elif num_layers == 98:
        init_block_channels = 96
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 160
        groups = 40
        k_sec = (3, 6, 20, 3)
        incs = (16, 32, 32, 128)
        test_time_pool = True
    elif num_layers == 107:
        init_block_channels = 128
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 200
        groups = 50
        k_sec = (4, 8, 20, 3)
        incs = (20, 64, 64, 128)
        test_time_pool = True
    elif num_layers == 131:
        init_block_channels = 128
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 160
        groups = 40
        k_sec = (4, 8, 28, 3)
        incs = (16, 32, 32, 128)
        test_time_pool = True
    else:
        raise ValueError("Unsupported DPN version with number of layers {}".format(num_layers))

    channels = [[0] * li for li in k_sec]
    rs = [0 * li for li in k_sec]
    bws = [0 * li for li in k_sec]
    for i in range(len(k_sec)):
        rs[i] = (2 ** i) * k_r
        bws[i] = (2 ** i) * 64 * bw_factor
        inc = incs[i]
        channels[i][0] = bws[i] + 3 * inc
        for j in range(1, k_sec[i]):
            channels[i][j] = channels[i][j - 1] + inc

    net = DPN(
        channels=channels,
        init_block_channels=init_block_channels,
        init_block_kernel_size=init_block_kernel_size,
        init_block_padding=init_block_padding,
        rs=rs,
        bws=bws,
        incs=incs,
        groups=groups,
        b_case=b_case,
        for_training=for_training,
        test_time_pool=test_time_pool,
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


def dpn68(**kwargs):
    """
    DPN-68 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=68, b_case=False, model_name="dpn68", **kwargs)


def dpn68b(**kwargs):
    """
    DPN-68b model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=68, b_case=True, model_name="dpn68b", **kwargs)


def dpn98(**kwargs):
    """
    DPN-98 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=98, b_case=False, model_name="dpn98", **kwargs)


def dpn107(**kwargs):
    """
    DPN-107 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=107, b_case=False, model_name="dpn107", **kwargs)


def dpn131(**kwargs):
    """
    DPN-131 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=131, b_case=False, model_name="dpn131", **kwargs)


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
    for_training = False

    models = [
        dpn68,
        # dpn68b,
        dpn98,
        # dpn107,
        dpn131,
    ]

    for model in models:

        net = model(pretrained=pretrained, for_training=for_training)

        net.train()
        # net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dpn68 or weight_count == 12611602)
        assert (model != dpn68b or weight_count == 12611602)
        assert (model != dpn98 or weight_count == 61570728)
        assert (model != dpn107 or weight_count == 86917800)
        assert (model != dpn131 or weight_count == 79254504)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
