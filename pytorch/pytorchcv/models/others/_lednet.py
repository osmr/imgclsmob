"""
    LEDNet for image segmentation, implemented in PyTorch.
    Original paper: 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1905.02423.
"""

__all__ = ['LEDNet', 'oth_lednet_cityscapes']

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, conv5x5_block, conv7x7_block, NormActivation,\
    InterpolationBlock, Hourglass, BreakBlock


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2


def channel_shuffle(x,
                    groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batchsize, -1, height, width)
    
    return x
    

class DownBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 bn_eps=1e-3):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.conv = conv3x3(
            in_channels=in_channel,
            out_channels=(out_channel - in_channel),
            stride=2,
            bias=True)
        self.norm_activ1 = NormActivation(
            in_channels=out_channel,
            bn_eps=bn_eps)

    def forward(self, x):
        y1 = self.pool(x)
        y2 = self.conv(x)

        diff_h = y2.size()[2] - y1.size()[2]
        diff_w = y2.size()[3] - y1.size()[3]
        y1 = F.pad(y1, pad=(diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))

        x = torch.cat((y2, y1), dim=1)
        x = self.norm_activ1(x)
        return x


class SsNbtBlock(nn.Module):
    def __init__(self,
                 chann,
                 dropprob,
                 dilated,
                 bn_eps=1e-3):
        super().__init__()

        oup_inc = chann // 2
        
        # dw
        self.conv3x1_1_l = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            bias=True)

        self.conv1x3_1_l = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=bn_eps)

        self.conv3x1_2_l = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1))

        self.conv1x3_2_l = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=bn_eps)
        
        # dw
        self.conv3x1_1_r = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            bias=True)

        self.conv1x3_1_r = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=bn_eps)

        self.conv3x1_2_r = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1))

        self.conv1x3_2_r = nn.Conv2d(
            in_channels=oup_inc,
            out_channels=oup_inc,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=bn_eps)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)       
        
    def forward(self, x):
        identity = x
        x1, x2 = split(x)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
    
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if self.dropout.p != 0:
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        x = torch.cat((output1, output2), dim=1)
        x = x + identity
        x = F.relu(x, inplace=True)
        x = channel_shuffle(x, groups=2)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 num_classes,
                 bn_eps=1e-3):
        super().__init__()
        self.initial_block = DownBlock(3, 32)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(SsNbtBlock(32, 0.03, 1))
        
        self.layers.append(DownBlock(32, 64))

        for x in range(0, 2):
            self.layers.append(SsNbtBlock(64, 0.03, 1))
  
        self.layers.append(DownBlock(64, 128))

        for x in range(0, 1):    
            self.layers.append(SsNbtBlock(128, 0.3, 1))
            self.layers.append(SsNbtBlock(128, 0.3, 2))
            self.layers.append(SsNbtBlock(128, 0.3, 5))
            self.layers.append(SsNbtBlock(128, 0.3, 9))
            
        for x in range(0, 1):    
            self.layers.append(SsNbtBlock(128, 0.3, 2))
            self.layers.append(SsNbtBlock(128, 0.3, 5))
            self.layers.append(SsNbtBlock(128, 0.3, 9))
            self.layers.append(SsNbtBlock(128, 0.3, 17))
                    
    def forward(self, x):
        x = self.initial_block(x)
        for layer in self.layers:
            x = layer(x)
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
        down_seq.add_module("down3", nn.Sequential(
            conv3x3_block(
                in_channels=att_out_channels,
                out_channels=att_out_channels,
                stride=2,
                bias=True,
                bn_eps=bn_eps),
            conv3x3_block(
                in_channels=att_out_channels,
                out_channels=att_out_channels,
                bias=True,
                bn_eps=bn_eps)
        ))

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
    def __init__(self,
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

        self.encoder = Encoder(
            num_classes,
            bn_eps=bn_eps)
        self.apn = APN(
            in_channels=128,
            out_channels=num_classes,
            bn_eps=bn_eps,
            in_size=(in_size[0] // 8, in_size[1] // 8))
        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.apn(x)
        x = self.up(x)
        return x


def oth_lednet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    bn_eps = 1e-3
    net = LEDNet(num_classes=num_classes, bn_eps=bn_eps)
    return net


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False

    in_size = (1024, 2048)

    models = [
        oth_lednet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_lednet_cityscapes or weight_count == 922821)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
