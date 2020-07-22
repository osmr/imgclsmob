
__all__ = ['oth_naivenet25']


import torch.nn as nn
from .common import conv1x1_block


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=True)


class Resv2Block(nn.Module):
    """ResNet v2 block without bn"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 is_branch=False):
        super(Resv2Block, self).__init__()
        self.is_branch = is_branch
        self.relu1 = nn.ReLU()
        self.conv1 = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=stride)

    def forward(self, x):
        out_branch = self.relu1(x)
        out = self.conv1(out_branch)
        out = self.relu2(out)
        out = self.conv2(out)
        out += x
        if self.is_branch:
            return out, out_branch
        else:
            return out


class BranchBlock(nn.Module):
    """
    Simple branch block.

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
        super(BranchBlock, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels,
            bias=True,
            use_bn=False)
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BranchUnit(nn.Module):
    """
    Simple branch unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    """
    def __init__(self,
                 in_channels,
                 mid_channels):
        super(BranchUnit, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            use_bn=False)
        self.conv_score = BranchBlock(
            in_channels=mid_channels,
            out_channels=2)
        self.conv_bbox = BranchBlock(
            in_channels=mid_channels,
            out_channels=4)

    def forward(self, x):
        out = self.conv1(x)
        out_score = self.conv_score(out)
        out_bbox = self.conv_bbox(out)
        return out_score, out_bbox


class NaiveNet(nn.Module):
    """NaiveNet for Fast Single Class Object Detection. 
    The entire backbone and branches only consists of conv 3×3, 
    conv 1×1, ReLU and residual connection.
    """
    def __init__(self):
        super(NaiveNet, self).__init__()
        self.block = Resv2Block

        num_filters_list = [32, 64, 128, 256]
        layers = [4, 2, 1, 3]

        self.conv1 = conv3x3(
            in_channels=3,
            out_channels=num_filters_list[1],
            stride=2,
            padding=0)
        self.relu1 = nn.ReLU()

        self.stage1_1 = self._make_layer(
            num_filters_list[1],
            num_filters_list[1],
            blocks=layers[0] - 1)

        self.stage1_2_branch1 = nn.Sequential(self.block(
            num_filters_list[1],
            num_filters_list[1],
            stride=1,
            is_branch=True))
        self.branch1 = nn.Sequential(BranchUnit(
            num_filters_list[1],
            num_filters_list[2]))

        self.stage1_3_branch2 = nn.Sequential(nn.ReLU())
        self.branch2 = nn.Sequential(BranchUnit(
            num_filters_list[1],
            num_filters_list[2]))

        self.stage2_1 = nn.Sequential(
            conv3x3(
                num_filters_list[1],
                num_filters_list[1],
                stride=2,
                padding=0),
            Resv2Block(
                num_filters_list[1],
                num_filters_list[1],
                stride=1,
                is_branch=False))

        self.stage2_2_branch3 = nn.Sequential(Resv2Block(
            num_filters_list[1],
            num_filters_list[1],
            stride=1,
            is_branch=True))
        self.branch3 = nn.Sequential(BranchUnit(
            num_filters_list[1],
            num_filters_list[2]))

        self.stage2_3_branch4 = nn.Sequential(nn.ReLU())
        self.branch4 = nn.Sequential(BranchUnit(
            num_filters_list[1],
            num_filters_list[2]))

        self.stage3_1 = nn.Sequential(
            conv3x3(
                num_filters_list[1],
                num_filters_list[2],
                stride=2,
                padding=0),
            Resv2Block(
                num_filters_list[2],
                num_filters_list[2],
                stride=1,
                is_branch=False))
        self.stage3_2_branch5 = nn.Sequential(nn.ReLU())
        self.branch5 = nn.Sequential(BranchUnit(
            num_filters_list[2],
            num_filters_list[2]))

        self.stage4_1 = nn.Sequential(
            conv3x3(
                num_filters_list[2],
                num_filters_list[2],
                stride=2,
                padding=0),
            Resv2Block(
                num_filters_list[2],
                num_filters_list[2],
                stride=1,
                is_branch=False))
        self.stage4_2_branch6 = nn.Sequential(Resv2Block(
            num_filters_list[2],
            num_filters_list[2],
            stride=1,
            is_branch=True))
        self.branch6 = nn.Sequential(BranchUnit(
            num_filters_list[2],
            num_filters_list[2]))

        self.stage4_3_branch7 = nn.Sequential(Resv2Block(
            num_filters_list[2],
            num_filters_list[2],
            stride=1,
            is_branch=True))
        self.branch7 = nn.Sequential(BranchUnit(
            num_filters_list[2],
            num_filters_list[2]))

        self.stage4_4_branch8 = nn.Sequential(nn.ReLU())
        self.branch8 = nn.Sequential(BranchUnit(
            num_filters_list[2],
            num_filters_list[2]))

    def _make_layer(self,
                    in_channels,
                    out_channels,
                    blocks):
        stride = 2
        layers = []
        layers.append(conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=0))
        for _ in range(blocks):
            layers.append(self.block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.stage1_1(x)

        x, b1 = self.stage1_2_branch1(x)
        score1, bbox1 = self.branch1(b1)

        x = b2 = self.stage1_3_branch2(x)
        score2, bbox2 = self.branch2(b2)

        x = self.stage2_1(x)
        x, b3 = self.stage2_2_branch3(x)
        score3, bbox3 = self.branch3(b3)

        x = b4 = self.stage2_3_branch4(x)
        score4, bbox4 = self.branch4(b4)

        x = self.stage3_1(x)
        x = b5 = self.stage3_2_branch5(x)
        score5, bbox5 = self.branch5(b5)

        x = self.stage4_1(x)
        x, b6 = self.stage4_2_branch6(x)
        score6, bbox6 = self.branch6(b6)

        x, b7 = self.stage4_3_branch7(x)
        score7, bbox7 = self.branch7(b7)

        x = b8 = self.stage4_4_branch8(x)
        score8, bbox8 = self.branch8(b8)

        outs = [score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5, score6, bbox6, score7,
                bbox7, score8, bbox8]
        return outs


def get_naivenet(arch, block, layers, pretrained=False, **kwargs):
    model = NaiveNet()
    return model


def oth_naivenet25(**kwargs):
    r"""NaiveNet-25 model from
    `"LFFD: A Light and Fast Face Detector for Edge Devices" <https://arxiv.org/pdf/1904.10633.pdf>`_
    It corresponds to the network structure built by `symbol_10_560_25L_8scales_v1.py` of mxnet version.
    """
    return get_naivenet(arch='naivenet25', block=Resv2Block, layers=[4, 2, 1, 3], **kwargs)


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
        oth_naivenet25,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net.train()
        net.eval()

        x = torch.randn(14, 3, 640, 640)
        y = net(x)
        # y.sum().backward()
        # assert (tuple(y.size()) == (14, 19, 1024, 2048))

        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_naivenet25 or weight_count == 2290608)


if __name__ == "__main__":
    _test()
