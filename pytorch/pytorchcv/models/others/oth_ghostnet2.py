"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import torch
import torch.nn as nn
import math
from oth_common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block,\
    dwsconv3x3_block, SEBlock


__all__ = ['oth_ghostnet']


class GhostHSigmoid(nn.Module):
    """
    Approximated sigmoid function, specific for GhostNet.
    """
    def forward(self, x):
        return torch.clamp(x, min=0.0, max=1.0)


class GhostConvBlock(nn.Module):
    """
    GhostNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(GhostConvBlock, self).__init__()
        pimary_out_channels = math.ceil(0.5 * out_channels)
        cheap_out_channels = out_channels - pimary_out_channels

        self.main_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=pimary_out_channels,
            activation=activation)
        self.cheap_conv = dwconv3x3_block(
            in_channels=pimary_out_channels,
            out_channels=cheap_out_channels,
            activation=activation)

    def forward(self, x):
        x = self.main_conv(x)
        y = self.cheap_conv(x)
        return torch.cat((x, y), dim=1)


class GhostExpBlock(nn.Module):
    """
    GhostNet expansion block for residual path in GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_kernel3,
                 exp_factor,
                 use_se):
        super(GhostExpBlock, self).__init__()
        self.use_dw_conv = (stride != 1)
        self.use_se = use_se
        mid_channels = int(math.ceil(exp_factor * in_channels))

        self.exp_conv = GhostConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels)
        if self.use_dw_conv:
            dw_conv_class = dwconv3x3_block if use_kernel3 else dwconv5x5_block
            self.dw_conv = dw_conv_class(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=None)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=4,
                out_activation=GhostHSigmoid())
        self.pw_conv = GhostConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.exp_conv(x)
        if self.use_dw_conv:
            x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class GhostUnit(nn.Module):
    """
    GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_kernel3,
                 exp_factor,
                 use_se):
        super(GhostUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = GhostExpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            use_kernel3=use_kernel3,
            exp_factor=exp_factor,
            use_se=use_se)
        if self.resize_identity:
            self.identity_conv = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                pw_activation=None)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class GhostNet(nn.Module):
    def __init__(self,
                 cfgs,
                 num_classes=1000,
                 width_mult=1.0):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        out_channels = round_channels(16 * width_mult, 4)
        layers = [
            conv3x3_block(
                in_channels=3,
                out_channels=out_channels,
                stride=2)
        ]
        in_channels = out_channels

        # building inverted residual blocks
        # block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            out_channels = round_channels(c * width_mult, 4)
            hidden_channel = round_channels(exp_size * width_mult, 4)
            exp_factor = (hidden_channel / in_channels)
            # print(exp_factor)
            layers.append(GhostUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=s,
                use_kernel3=(k == 3),
                exp_factor=exp_factor,
                use_se=(use_se == 1)))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # building last several layers
        out_channels = round_channels(exp_size * width_mult, 4)
        print(out_channels)
        self.squeeze = nn.Sequential(
            conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        in_channels = out_channels

        out_channels = 1280
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def oth_ghostnet(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    assert (pretrained is not None)
    cfgs = [
        # k, t, c, SE, s 
        [3,  16,  16, 0, 1],

        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],

        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],

        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],

        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


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
        oth_ghostnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_ghostnet or weight_count == 5180840)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
