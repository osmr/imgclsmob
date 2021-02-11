"""
    LEDNet for image segmentation, implemented in PyTorch.
    Original paper: 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.
"""

__all__ = ['LEDNet', 'lednet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1_block, conv3x3_block, conv5x5_block, conv7x7_block, asym_conv3x3_block, ChannelShuffle,\
    InterpolationBlock, Hourglass, BreakBlock
from .enet import ENetMixDownBlock


class LEDBranch(nn.Module):
    """
    LEDNet encoder branch.

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
        super(LEDBranch, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = asym_conv3x3_block(
            channels=channels,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps)
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


class LEDUnit(nn.Module):
    """
    LEDNet encoder unit (Split-Shuffle-non-bottleneck).

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
        super(LEDUnit, self).__init__()
        mid_channels = channels // 2

        self.left_branch = LEDBranch(
            channels=mid_channels,
            dilation=dilation,
            dropout_rate=dropout_rate,
            bn_eps=bn_eps)
        self.right_branch = LEDBranch(
            channels=mid_channels,
            dilation=dilation,
            dropout_rate=dropout_rate,
            bn_eps=bn_eps)
        self.activ = nn.ReLU(inplace=True)
        self.shuffle = ChannelShuffle(
            channels=channels,
            groups=2)

    def forward(self, x):
        identity = x

        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.left_branch(x1)
        x2 = self.right_branch(x2)
        x = torch.cat((x1, x2), dim=1)

        x = x + identity
        x = self.activ(x)
        x = self.shuffle(x)
        return x


class PoolingBranch(nn.Module):
    """
    Pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    bn_eps : float
        Small float added to variance in Batch norm.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    down_size : int
        Spatial size of downscaled image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias,
                 bn_eps,
                 in_size,
                 down_size):
        super(PoolingBranch, self).__init__()
        self.in_size = in_size

        self.pool = nn.AdaptiveAvgPool2d(output_size=down_size)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            bn_eps=bn_eps)
        self.up = InterpolationBlock(
            scale_factor=None,
            out_size=in_size)

    def forward(self, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        x = self.up(x, in_size)
        return x


class APN(nn.Module):
    """
    Attention pyramid network block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 in_size):
        super(APN, self).__init__()
        self.in_size = in_size
        att_out_channels = 1

        self.pool_branch = PoolingBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            bn_eps=bn_eps,
            in_size=in_size,
            down_size=1)

        self.body = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            bn_eps=bn_eps)

        down_seq = nn.Sequential()
        down_seq.add_module("down1", conv7x7_block(
            in_channels=in_channels,
            out_channels=att_out_channels,
            stride=2,
            bias=True,
            bn_eps=bn_eps))
        down_seq.add_module("down2", conv5x5_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            stride=2,
            bias=True,
            bn_eps=bn_eps))
        down3_subseq = nn.Sequential()
        down3_subseq.add_module("conv1", conv3x3_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            stride=2,
            bias=True,
            bn_eps=bn_eps))
        down3_subseq.add_module("conv2", conv3x3_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            bias=True,
            bn_eps=bn_eps))
        down_seq.add_module("down3", down3_subseq)

        up_seq = nn.Sequential()
        up = InterpolationBlock(scale_factor=2)
        up_seq.add_module("up1", up)
        up_seq.add_module("up2", up)
        up_seq.add_module("up3", up)

        skip_seq = nn.Sequential()
        skip_seq.add_module("skip1", BreakBlock())
        skip_seq.add_module("skip2", conv7x7_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            bias=True,
            bn_eps=bn_eps))
        skip_seq.add_module("skip3", conv5x5_block(
            in_channels=att_out_channels,
            out_channels=att_out_channels,
            bias=True,
            bn_eps=bn_eps))

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq)

    def forward(self, x):
        y = self.pool_branch(x)
        w = self.hg(x)
        x = self.body(x)
        x = x * w
        x = x + y
        return x


class LEDNet(nn.Module):
    """
    LEDNet model from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit.
    dilations : list of int
        Dilations for units.
    dropout_rates : list of list of int
        Dropout rates for each unit in encoder.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images in encoder.
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
                 dropout_rates,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(LEDNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        self.encoder = nn.Sequential()
        for i, dilations_per_stage in enumerate(dilations):
            out_channels = channels[i]
            dropout_rate = dropout_rates[i]
            stage = nn.Sequential()
            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), ENetMixDownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=True,
                        bn_eps=bn_eps,
                        correct_size_mismatch=correct_size_mismatch))
                    in_channels = out_channels
                else:
                    stage.add_module("unit{}".format(j + 1), LEDUnit(
                        channels=in_channels,
                        dilation=dilation,
                        dropout_rate=dropout_rate,
                        bn_eps=bn_eps))
            self.encoder.add_module("stage{}".format(i + 1), stage)
        self.apn = APN(
            in_channels=in_channels,
            out_channels=num_classes,
            bn_eps=bn_eps,
            in_size=(in_size[0] // 8, in_size[1] // 8) if fixed_size else None)
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
        x = self.encoder(x)
        x = self.apn(x)
        x = self.up(x)
        return x


def get_lednet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create LEDNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [32, 64, 128]
    dilations = [[0, 1, 1, 1], [0, 1, 1], [0, 1, 2, 5, 9, 2, 5, 9, 17]]
    dropout_rates = [0.03, 0.03, 0.3]
    bn_eps = 1e-3

    net = LEDNet(
        channels=channels,
        dilations=dilations,
        dropout_rates=dropout_rates,
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


def lednet_cityscapes(num_classes=19, **kwargs):
    """
    LEDNet model for Cityscapes from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic
    Segmentation,' https://arxiv.org/abs/1905.02423.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_lednet(num_classes=num_classes, model_name="lednet_cityscapes", **kwargs)


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
    correct_size_mismatch = False
    in_size = (1024, 2048)
    classes = 19

    models = [
        lednet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size,
                    correct_size_mismatch=correct_size_mismatch)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lednet_cityscapes or weight_count == 922821)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
