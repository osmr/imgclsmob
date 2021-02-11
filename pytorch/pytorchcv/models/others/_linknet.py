"""
    LinkNet for image segmentation, implemented in PyTorch.
    Original paper: 'LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation,'
    https://arxiv.org/abs/1707.03718.
"""

__all__ = ['LinkNet', 'oth_linknet_cityscapes']

import torch
import torch.nn as nn
from torchvision.models import resnet
from common import conv1x1_block, deconv3x3_block, Hourglass, Identity


class DecoderStage(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 output_padding,
                 bias):
        super(DecoderStage, self).__init__()
        mid_channels = in_channels // 4

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias)
        self.conv2 = deconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            out_padding=output_padding,
            bias=bias)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LinkNetHead(nn.Module):
    """
    LinkNet head block.

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
        super(LinkNetHead, self).__init__()
        mid_channels = in_channels // 2

        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(mid_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        x = self.tp_conv1(x)
        x = self.conv2(x)
        x = self.tp_conv2(x)
        return x


class LinkNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        bias = False
        backbone = resnet.resnet18(pretrained=False)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        in_channels = 512
        channels = [256, 128, 64, 64]
        strides = [2, 2, 2, 1]
        output_paddings = [1, 1, 1, 0]

        down_seq = nn.Sequential()
        down_seq.add_module("down1", backbone.layer1)
        down_seq.add_module("down2", backbone.layer2)
        down_seq.add_module("down3", backbone.layer3)
        down_seq.add_module("down4", backbone.layer4)

        up_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i, out_channels in enumerate(channels):
            up_seq.add_module("up{}".format(i + 1), DecoderStage(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides[i],
                output_padding=output_paddings[i],
                bias=bias))
            in_channels = out_channels
            skip_seq.add_module("skip{}".format(i + 1), Identity())
        up_seq = up_seq[::-1]

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq)

        self.head = LinkNetHead(
            in_channels=in_channels,
            out_channels=num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.hg(x)
        x = self.head(x)
        return x


def oth_linknet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return LinkNet(num_classes=num_classes, **kwargs)


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
        oth_linknet_cityscapes,
    ]

    for model in models:

        # from torchsummary import summary
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # net = LinkNet(num_classes=19).to(device)
        # summary(net, (3, 512, 1024))

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_linknet_cityscapes or weight_count == 11535699)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()

