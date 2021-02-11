"""
    ERFNet for image segmentation, implemented in PyTorch.
    Original paper: 'ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation,'
    http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf.
"""

__all__ = ['ERFNet', 'oth_erfnet_cityscapes']

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import deconv3x3_block, AsymConvBlock
# from common import conv3x3, NormActivation
from enet import ENetMixDownBlock


class FCU(nn.Module):
    """
    Factorized convolution unit.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(FCU, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        padding1 = (kernel_size - 1) // 2
        padding2 = padding1 * dilation

        self.conv1 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding1,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps)
        self.conv2 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding2,
            dilation=dilation,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)

        x = x + identity
        x = self.activ(x)
        return x


class ERFNetUnit(nn.Module):
    def __init__(self,
                 channels,
                 dropout_rate,
                 dilation):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(channels, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1 * dilation, 0), bias=True, dilation = (dilation, 1))

        self.conv1x3_2 = nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, 1 * dilation), bias=True, dilation = (1, dilation))

        self.bn2 = nn.BatchNorm2d(channels, eps=1e-03)

        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):

        output = self.conv3x1_1(x)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output + x)    #+input = identity (residual connection)


class ERFNet(nn.Module):
    def __init__(self,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super().__init__()

        downs = [1, 1, 1, 0, 0]
        channels = [16, 64, 128, 64, 16]
        dilations = [[1], [1, 1, 1, 1, 1, 1], [1, 2, 4, 8, 16, 2, 4, 8, 16], [1, 1, 1], [1, 1, 1]]
        dropout_rates = [[0.0], [0.03, 0.03, 0.03, 0.03, 0.03, 0.03], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        enc_idx = 0
        dec_idx = 0
        for i, out_channels in enumerate(channels):
            dilations_per_stage = dilations[i]
            dropout_rates_per_stage = dropout_rates[i]
            is_down = downs[i]
            stage = nn.Sequential()
            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    if is_down:
                        unit = ENetMixDownBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bias=True,
                            bn_eps=bn_eps,
                            correct_size_mismatch=correct_size_mismatch)
                    else:
                        unit = deconv3x3_block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            stride=2,
                            bias=True,
                            bn_eps=bn_eps)
                else:
                    unit = FCU(
                        channels=in_channels,
                        kernel_size=3,
                        dilation=dilation,
                        dropout_rate=dropout_rates_per_stage[j],
                        bn_eps=bn_eps)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            if is_down:
                enc_idx += 1
                self.encoder.add_module("stage{}".format(enc_idx), stage)
            else:
                dec_idx += 1
                self.decoder.add_module("stage{}".format(dec_idx), stage)

        self.head = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x


def oth_erfnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return ERFNet(num_classes=num_classes, **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    # fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        oth_erfnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_erfnet_cityscapes or weight_count == 2064191)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
