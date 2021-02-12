"""
    SQNet for image segmentation, implemented in PyTorch.
    Original paper: 'Speeding up Semantic Segmentation for Autonomous Driving,'
    https://https://openreview.net/pdf?id=S1uHiFyyg.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import conv1x1, conv1x1_block, conv3x3_block, InterpolationBlock, Hourglass, Identity, Concurrent


class FireBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 squeeze_channels,
                 expand_channels,
                 bias=True,
                 use_bn=False,
                 activation=(lambda: nn.ELU(inplace=True))):
        super(FireBlock, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            bias=bias,
            use_bn=use_bn,
            activation=activation)
        self.branches = Concurrent(merge_type="cat")
        self.branches.add_module("branch1", conv1x1_block(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            bias=bias,
            use_bn=use_bn,
            activation=None))
        self.branches.add_module("branch2", conv3x3_block(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            bias=bias,
            use_bn=use_bn,
            activation=None))
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.branches(x)
        x = self.activ(x)
        return x


class ParallelDilatedConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 use_bn=False,
                 activation=(lambda: nn.ELU(inplace=True))):
        super(ParallelDilatedConv, self).__init__()
        dilations = [1, 2, 3, 4]

        self.branches = Concurrent(merge_type="sum")
        for i, dilation in enumerate(dilations):
            self.branches.add_module("branch{}".format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=dilation,
                dilation=dilation,
                bias=bias,
                use_bn=use_bn,
                activation=activation))

    def forward(self, x):
        x = self.branches(x)
        return x


class SQNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1) # 32
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16

        self.fire1_1 = FireBlock(96, 16, 64)
        self.fire1_2 = FireBlock(128, 16, 64)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire2_1 = FireBlock(128, 32, 128)
        self.fire2_2 = FireBlock(256, 32, 128)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire3_1 = FireBlock(256, 64, 256)
        self.fire3_2 = FireBlock(512, 64, 256)
        self.fire3_3 = FireBlock(512, 64, 256)
        self.parallel = ParallelDilatedConv(512, 512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ELU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        self.relu3 = nn.ELU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(256, 96, 3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ELU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(192, self.num_classes, 3, stride=2, padding=1, output_padding=1)

        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv1_1 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1) # 32
        self.conv1_2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1) # 32

        self.relu1_1 = nn.ELU(inplace=True)
        self.relu1_2 = nn.ELU(inplace=True)
        self.relu2_1 = nn.ELU(inplace=True)
        self.relu2_2 = nn.ELU(inplace=True)
        self.relu3_1 = nn.ELU(inplace=True)
        self.relu3_2 = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x_1 = self.relu1(x)

        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)

        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)

        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        x = self.fire3_3(x)
        x = self.parallel(x)

        y_3 = self.deconv1(x)
        y_3 = self.relu2(y_3)

        x_3 = self.conv3_1(x_3)
        x_3 = self.relu3_1(x_3)
        x_3 = F.interpolate(x_3, y_3.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x_3, y_3], 1)
        x = self.conv3_2(x)
        x = self.relu3_2(x)

        # concat x_3
        y_2 = self.deconv2(x)
        y_2 = self.relu3(y_2)

        x_2 = self.conv2_1(x_2)
        x_2 = self.relu2_1(x_2)
        # concat x_2
        y_2 = F.interpolate(y_2, x_2.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x_2, y_2], 1)
        x = self.conv2_2(x)
        x = self.relu2_2(x)

        y_1 = self.deconv3(x)
        y_1 = self.relu4(y_1)

        x_1 = self.conv1_1(x_1)
        x_1 = self.relu1_1(x_1)
        x = torch.cat([x_1, y_1], 1)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.deconv4(x)
        return x


def oth_sqnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return SQNet(num_classes=num_classes, **kwargs)


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
        oth_sqnet_cityscapes,
    ]

    for model in models:

        # from torchsummary import summary
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # net = SQNet(num_classes=19).to(device)
        # summary(net, (3, 512, 1024))

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_sqnet_cityscapes or weight_count == 16262771)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
