"""
    ENet for image segmentation, implemented in PyTorch.
    Original paper: 'ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1606.02147.
"""

import torch
import torch.nn as nn
from common import ConvBlock, AsymConvBlock, DeconvBlock, NormActivation, conv1x1_block


class InitBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 bias,
                 bn_eps,
                 activation):
        super(InitBlock, self).__init__()
        self.main_branch = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(out_channels - in_channels),
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias)
        self.ext_branch = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=2,
            padding=padding)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=activation)

    def forward(self, x):
        x1 = self.main_branch(x)
        x2 = self.ext_branch(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.norm_activ(x)
        return x


class DownBlock(nn.Module):
    def __init__(self,
                 ext_channels,
                 kernel_size,
                 padding):
        super().__init__()
        self.ext_channels = ext_channels

        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            return_indices=True)

    def forward(self, x):
        x, max_indices = self.pool(x)
        branch, _, height, width = x.size()
        pad = torch.zeros(branch, self.ext_channels, height, width, dtype=x.dtype, device=x.device)
        x = torch.cat((x, pad), dim=1)
        return x, max_indices


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super().__init__()
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


class ENetUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 dilation,
                 use_asym_conv,
                 dropout_rate,
                 bias,
                 activation,
                 down,
                 bottleneck_factor=4):
        super().__init__()
        self.resize_identity = (in_channels != out_channels)
        self.down = down
        mid_channels = in_channels // bottleneck_factor

        if not self.resize_identity:
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bias=bias,
                activation=activation)
            if use_asym_conv:
                self.conv2 = AsymConvBlock(
                    channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    lw_activation=activation,
                    rw_activation=activation)
            else:
                self.conv2 = ConvBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    activation=activation)
        elif self.down:
            self.identity_block = DownBlock(
                ext_channels=(out_channels - in_channels),
                kernel_size=kernel_size,
                padding=padding)
            self.conv1 = ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=1,
                bias=bias,
                activation=activation)
            self.conv2 = ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=bias,
                activation=activation)
        else:
            self.identity_block = UpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias)
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bias=bias,
                activation=activation)
            self.conv2 = DeconvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                out_padding=1,
                dilation=dilation,
                bias=bias,
                activation=activation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            activation=activation)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.activ = activation()

    def forward(self, x, max_indices=None):
        if not self.resize_identity:
            identity = x
        elif self.down:
            identity, max_indices = self.identity_block(x)
        else:
            identity = self.identity_block(x, max_indices)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = x + identity
        x = self.activ(x)

        if self.resize_identity and self.down:
            return x, max_indices
        else:
            return x
        # return x, max_indices


class ENetStage(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 dilation,
                 use_asym_conv,
                 dropout_rate,
                 bias,
                 activation,
                 down,
                 layers):
        super().__init__()
        self.scale_unit = ENetUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            use_asym_conv=use_asym_conv,
            dropout_rate=dropout_rate,
            bias=bias,
            activation=activation,
            down=down)

        self.units = nn.Sequential()
        for i in range(1, layers):
            self.units.add_module("unit{}".format(i + 1), ENetUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                use_asym_conv=use_asym_conv,
                dropout_rate=dropout_rate,
                bias=bias,
                activation=activation,
                down=down))

    def forward(self, x):
        x, max_indices = self.scale_unit(x)
        x = self.units(x)
        return x, max_indices


class ENet(nn.Module):
    def __init__(self,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super().__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        bias = False
        encoder_activation = (lambda: nn.PReLU(1))
        decoder_activation = (lambda: nn.ReLU(inplace=True))

        out_channels = 16
        self.initial_block = InitBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            bn_eps=bn_eps,
            activation=encoder_activation)
        in_channels = out_channels

        layers = [5]
        channels = [64]
        kernel_sizes = [3]
        dilations = [1]
        use_asym_convs = [0]
        dropout_rates = [0.01]

        self.stage1 = ENetStage(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=kernel_sizes[0],
            padding=dilations[0],
            dilation=dilations[0],
            use_asym_conv=(use_asym_convs[0] == 1),
            dropout_rate=dropout_rates[0],
            bias=bias,
            activation=encoder_activation,
            down=True,
            layers=layers[0])

        # # Stage 1 - Encoder
        # self.downsample1_0 = ENetUnit(
        #     in_channels=16,
        #     out_channels=64,
        #     kernel_size=3,
        #     padding=1,
        #     dilation=1,
        #     use_asym_conv=False,
        #     dropout_rate=0.01,
        #     bias=bias,
        #     activation=encoder_activation,
        #     down=True)
        # self.regular1_1 = ENetUnit(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     padding=1,
        #     dilation=1,
        #     use_asym_conv=False,
        #     dropout_rate=0.01,
        #     bias=bias,
        #     activation=encoder_activation,
        #     down=True)
        # self.regular1_2 = ENetUnit(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     padding=1,
        #     dilation=1,
        #     use_asym_conv=False,
        #     dropout_rate=0.01,
        #     bias=bias,
        #     activation=encoder_activation,
        #     down=True)
        # self.regular1_3 = ENetUnit(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     padding=1,
        #     dilation=1,
        #     use_asym_conv=False,
        #     dropout_rate=0.01,
        #     bias=bias,
        #     activation=encoder_activation,
        #     down=True)
        # self.regular1_4 = ENetUnit(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     padding=1,
        #     dilation=1,
        #     use_asym_conv=False,
        #     dropout_rate=0.01,
        #     bias=bias,
        #     activation=encoder_activation,
        #     down=True)

        # Stage 2 - Encoder
        self.downsample2_0 = ENetUnit(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.regular2_1 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated2_2 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=2,
            dilation=2,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.asymmetric2_3 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            padding=2,
            dilation=1,
            use_asym_conv=True,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated2_4 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=4,
            dilation=4,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.regular2_5 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated2_6 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=8,
            dilation=8,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.asymmetric2_7 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            padding=2,
            dilation=1,
            use_asym_conv=True,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated2_8 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=16,
            dilation=16,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)

        # Stage 3 - Encoder
        self.regular3_0 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated3_1 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            dilation=2,
            padding=2,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.asymmetric3_2 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            padding=2,
            dilation=1,
            use_asym_conv=True,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated3_3 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=4,
            dilation=4,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.regular3_4 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated3_5 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=8,
            dilation=8,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.asymmetric3_6 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            padding=2,
            dilation=1,
            use_asym_conv=True,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)
        self.dilated3_7 = ENetUnit(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=16,
            dilation=16,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=encoder_activation,
            down=True)

        # Stage 4 - Decoder
        self.upsample4_0 = ENetUnit(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=decoder_activation,
            down=False)
        self.regular4_1 = ENetUnit(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=decoder_activation,
            down=False)
        self.regular4_2 = ENetUnit(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=decoder_activation,
            down=False)

        # Stage 5 - Decoder
        self.upsample5_0 = ENetUnit(
            in_channels=64,
            out_channels=16,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=decoder_activation,
            down=False)
        self.regular5_1 = ENetUnit(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            dilation=1,
            use_asym_conv=False,
            dropout_rate=0.1,
            bias=bias,
            activation=decoder_activation,
            down=False)

        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

        self.project_layer = nn.Conv2d(
            128,
            num_classes,
            1,
            bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # # Stage 1 - Encoder
        # x, max_indices1_0 = self.downsample1_0(x)
        # x = self.regular1_1(x)
        # x = self.regular1_2(x)
        # x = self.regular1_3(x)
        # x = self.regular1_4(x)

        x, max_indices1_0 = self.stage1(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)

        return x


def oth_enet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return ENet(num_classes=num_classes, **kwargs)


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
        oth_enet_cityscapes,
    ]

    for model in models:

        # from torchsummary import summary
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = ENet(num_classes=19).to(device)
        # summary(model, (3, 512, 1024))

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != oth_enet_cityscapes or weight_count == 360422)
        assert (model != oth_enet_cityscapes or weight_count == 360492)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
