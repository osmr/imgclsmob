"""
    U-Net for image segmentation, implemented in PyTorch.
    Original paper: 'U-Net: Convolutional Networks for Biomedical Image Segmentation,'
    https://arxiv.org/abs/1505.04597.
"""

import torch
import torch.nn as nn
from common import conv1x1, conv3x3_block, InterpolationBlock, Hourglass, Identity


class UNetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDownStage(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetDownStage, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = UNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UNetUpStage(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetUpStage, self).__init__()
        self.conv = UNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)
        self.up = InterpolationBlock(
            scale_factor=2,
            align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class UNetHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetHead, self).__init__()
        self.conv1 = UNetBlock(
            in_channels=(2 * in_channels),
            out_channels=in_channels,
            bias=bias)
        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(UNet, self).__init__()
        bias = True

        init_block_channels = 64
        channels = [[128, 256, 512, 512], [512, 256, 128, 64]]

        self.stem = UNetBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bias=bias)
        in_channels = init_block_channels

        down_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[0]):
            down_seq.add_module("down{}".format(i + 1), UNetDownStage(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias))
            in_channels = out_channels
            skip_seq.add_module("skip{}".format(i + 1), Identity())

        up_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[1]):
            if i == 0:
                up_seq.add_module("down{}".format(i + 1), InterpolationBlock(
                    scale_factor=2,
                    align_corners=True))
            else:
                up_seq.add_module("down{}".format(i + 1), UNetUpStage(
                    in_channels=(2 * in_channels),
                    out_channels=out_channels,
                    bias=bias))
            in_channels = out_channels
        up_seq = up_seq[::-1]

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            merge_type="cat")

        self.head = UNetHead(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.hg(x)
        x = self.head(x)
        return x


def unet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return UNet(num_classes=num_classes, **kwargs)


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
        unet_cityscapes,
    ]

    for model in models:

        # from torchsummary import summary
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # net = UNet(num_classes=19).to(device)
        # summary(net, (3, 512, 1024))

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != unet_cityscapes or weight_count == 13396499)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
