"""
    LinkNet for image segmentation, implemented in PyTorch.
    Original paper: 'LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation,'
    https://arxiv.org/abs/1707.03718.
"""

__all__ = ['LinkNet', 'oth_linknet_cityscapes']

import torch
import torch.nn as nn
from torchvision.models import resnet


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    diffy = (h - max_height) // 2
    diffx = (w -max_width) // 2
    return layer[:,:,diffy:(diffy + max_height),diffx:(diffx + max_width)]


class DecoderStage(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 bias=False):
        super(DecoderStage, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                in_planes//4,
                1,
                1,
                0,
                bias=bias),
            nn.BatchNorm2d(in_planes//4),
            nn.ReLU(inplace=True))
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes//4,
                in_planes//4,
                kernel_size,
                stride,
                padding,
                output_padding,
                bias=bias),
            nn.BatchNorm2d(in_planes//4),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_planes//4,
                out_planes,
                1,
                1,
                0,
                bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x_high_level, x_low_level):
        x = self.conv1(x_high_level)
        x = self.tp_conv(x)

        x = center_crop(x, x_low_level.size()[2], x_low_level.size()[3])

        x = self.conv2(x)

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
    bias : bool
        Whether the layer uses a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(ENetUpBlock, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            activation=None)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, max_indices):
        x = self.conv(x)
        x = self.unpool(x, max_indices)
        return x


class LinkNet(nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()

        base = resnet.resnet18(pretrained=False)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = DecoderStage(64, 64, 3, 1, 1, 0)
        self.decoder2 = DecoderStage(128, 64, 3, 2, 1, 1)
        self.decoder3 = DecoderStage(256, 128, 3, 2, 1, 1)
        self.decoder4 = DecoderStage(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, num_classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y


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

