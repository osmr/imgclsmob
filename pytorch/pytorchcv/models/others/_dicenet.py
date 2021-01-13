import math
import torch
from torch.nn import init
from torch import nn
import torch.nn.functional as F
from common import conv1x1_block, conv3x3_block, NormActivation, ChannelShuffle


class DiceBlock(nn.Module):
    '''
    This class implements the volume-wise seperable convolutions
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size):
        super(DiceBlock, self).__init__()
        kernel_size = 3
        padding = 1
        height = in_size[0]
        width = in_size[1]

        self.conv_channel = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels,
            padding=padding,
            bias=False)
        self.conv_width = nn.Conv2d(
            width,
            width,
            kernel_size=kernel_size,
            stride=1,
            groups=width,
            padding=padding,
            bias=False)
        self.conv_height = nn.Conv2d(
            height,
            height,
            kernel_size=kernel_size,
            stride=1,
            groups=height,
            padding=padding,
            bias=False)

        self.br_act = NormActivation(
            in_channels=(3 * in_channels),
            activation=(lambda: nn.PReLU(3 * in_channels)))

        self.weight_avg_layer = conv1x1_block(
            in_channels=(3 * in_channels),
            out_channels=in_channels,
            groups=in_channels,
            activation=(lambda: nn.PReLU(in_channels)))

        # project from channel_in to Channel_out
        groups_proj = math.gcd(in_channels, out_channels)
        self.proj_layer = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups_proj,
            activation=(lambda: nn.PReLU(out_channels)))
        self.linear_comb_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.vol_shuffle = ChannelShuffle(
            channels=(3 * in_channels),
            groups=3)

        self.width = width
        self.height = height

    def forward(self, x):
        batch, channels, height, width = x.size()
        # process across channel. Input: C x H x W, Output: C x H x W
        out_ch_wise = self.conv_channel(x)

        # process across height. Input: H x C x W, Output: C x H x W
        x_h_wise = x.clone()
        if height != self.height:
            if height < self.height:
                x_h_wise = F.interpolate(x_h_wise, mode='bilinear', size=(self.height, width), align_corners=True)
            else:
                x_h_wise = F.adaptive_avg_pool2d(x_h_wise, output_size=(self.height, width))

        x_h_wise = x_h_wise.transpose(1, 2).contiguous()
        out_h_wise = self.conv_height(x_h_wise).transpose(1, 2).contiguous()

        h_wise_height = out_h_wise.size(2)
        if height != h_wise_height:
            if h_wise_height < height:
                out_h_wise = F.interpolate(out_h_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_h_wise = F.adaptive_avg_pool2d(out_h_wise, output_size=(height, width))

        # process across width: Input: W x H x C, Output: C x H x W
        x_w_wise = x.clone()
        if width != self.width:
            if width < self.width:
                x_w_wise = F.interpolate(x_w_wise, mode='bilinear', size=(height, self.width), align_corners=True)
            else:
                x_w_wise = F.adaptive_avg_pool2d(x_w_wise, output_size=(height, self.width))

        x_w_wise = x_w_wise.transpose(1, 3).contiguous()
        out_w_wise = self.conv_width(x_w_wise).transpose(1, 3).contiguous()
        w_wise_width = out_w_wise.size(3)
        if width != w_wise_width:
            if w_wise_width < width:
                out_w_wise = F.interpolate(out_w_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_w_wise = F.adaptive_avg_pool2d(out_w_wise, output_size=(height, width))

        # Merge. Output will be 3C x H X W
        outputs = torch.cat((out_ch_wise, out_h_wise, out_w_wise), 1)
        outputs = self.br_act(outputs)
        outputs = self.vol_shuffle(outputs)
        outputs = self.weight_avg_layer(outputs)
        linear_wts = self.linear_comb_layer(outputs)
        proj_out = self.proj_layer(outputs)
        return proj_out * linear_wts


class StridedDiceBlock(nn.Module):
    '''
    This class implements the strided volume-wise seperable convolutions
    '''
    def __init__(self,
                 in_channels,
                 in_size):
        super(StridedDiceBlock, self).__init__()

        self.left_layer = nn.Sequential(
            conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=2,
                groups=in_channels,
                activation=(lambda: nn.PReLU(in_channels))),
            conv1x1_block(
                in_channels=in_channels,
                out_channels=in_channels,
                activation=(lambda: nn.PReLU(in_channels)))
        )
        in_size = (in_size[0] // 2, in_size[1] // 2)
        self.right_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            DiceBlock(
                in_channels,
                in_channels,
                in_size=in_size),
            conv1x1_block(
                in_channels=in_channels,
                out_channels=in_channels,
                activation=(lambda: nn.PReLU(in_channels)))
        )
        self.shuffle = ChannelShuffle(
            channels=(2 * in_channels),
            groups=2)

    def forward(self, x):
        x_left = self.left_layer(x)
        x_right = self.right_layer(x)
        concat = torch.cat([x_left, x_right], 1)
        return self.shuffle(concat)


class ShuffledDiceBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 c_tag=0.5):
        super(ShuffledDiceBlock, self).__init__()
        self.left_part = round(c_tag * in_channels)
        right_in_channels = in_channels - self.left_part
        right_out_channels = out_channels - self.left_part

        self.layer_right = nn.Sequential(
            conv1x1_block(
                in_channels=right_in_channels,
                out_channels=right_out_channels,
                activation=(lambda: nn.PReLU(right_out_channels))),
            DiceBlock(
                in_channels=right_out_channels,
                out_channels=right_out_channels,
                in_size=in_size)
        )

        self.shuffle = ChannelShuffle(
            channels=(2 * right_out_channels),
            groups=2)

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]

        right = self.layer_right(right)

        return self.shuffle(torch.cat((left, right), 1))


class CNNModel(nn.Module):
    def __init__(self,
                 out_channel_map,
                 rep_layers,
                 drop_p,
                 num_classes=1000,
                 channels=3,
                 in_size=(224, 224)):
        super(CNNModel, self).__init__()
        assert (in_size[0] % 32 == 0)
        assert (in_size[1] % 32 == 0)

        channels_in = channels
        reps_at_each_level = rep_layers

        # output size will be 112 x 112
        self.level1 = conv3x3_block(
            in_channels=channels_in,
            out_channels=out_channel_map[0],
            stride=2,
            activation=(lambda: nn.PReLU(out_channel_map[0])))
        in_size = (in_size[0] // 2, in_size[1] // 2)

        self.level2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        in_size = (in_size[0] // 2, in_size[1] // 2)

        # output size will be 28 x 28
        level3 = nn.ModuleList()
        level3.append(StridedDiceBlock(in_channels=out_channel_map[1], in_size=in_size))
        in_size = (in_size[0] // 2, in_size[1] // 2)
        for i in range(reps_at_each_level[1]):
            if i == 0:
                level3.append(ShuffledDiceBlock(2 * out_channel_map[1], out_channel_map[2], in_size=in_size))
            else:
                level3.append(ShuffledDiceBlock(out_channel_map[2], out_channel_map[2], in_size=in_size))
        self.level3 = nn.Sequential(*level3)

        # output size will be 14 x 14
        level4 = nn.ModuleList()
        level4.append(StridedDiceBlock(in_channels=out_channel_map[2], in_size=in_size))
        in_size = (in_size[0] // 2, in_size[1] // 2)
        for i in range(reps_at_each_level[2]):
            if i == 0:
                level4.append(ShuffledDiceBlock(2 * out_channel_map[2], out_channel_map[3], in_size=in_size))
            else:
                level4.append(ShuffledDiceBlock(out_channel_map[3], out_channel_map[3], in_size=in_size))
        self.level4 = nn.Sequential(*level4)

        # output size will be 7 x 7
        level5 = nn.ModuleList()
        level5.append(StridedDiceBlock(in_channels=out_channel_map[3], in_size=in_size))
        in_size = (in_size[0] // 2, in_size[1] // 2)
        for i in range(reps_at_each_level[3]):
            if i == 0:
                level5.append(ShuffledDiceBlock(2 * out_channel_map[3], out_channel_map[4], in_size=in_size))
            else:
                level5.append(ShuffledDiceBlock(out_channel_map[4], out_channel_map[4], in_size=in_size))
        self.level5 = nn.Sequential(*level5)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        groups = 4
        self.classifier = nn.Sequential(
            nn.Conv2d(out_channel_map[4], out_channel_map[5], kernel_size=1, groups=groups, bias=False),
            nn.Dropout(p=drop_p),
            nn.Conv2d(out_channel_map[5], num_classes, 1, padding=0, bias=True)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


def get_dicenet(s):
    rep_layers = [0, 3, 7, 3]

    sc_ch_dict = {
        # 0.1 : [8, 8, 16, 32, 64, 512],
        0.2: [16, 16, 32, 64, 128, 1024],
        0.5: [24, 24, 48, 96, 192, 1024],
        0.75: [24, 24, 86, 172, 344, 1024],
        1.0: [24, 24, 116, 232, 464, 1024],
        1.25: [24, 24, 144, 288, 576, 1024],
        1.5: [24, 24, 176, 352, 704, 1024],
        1.75: [24, 24, 210, 420, 840, 1024],
        2.0: [24, 24, 244, 488, 976, 1024],
        2.4: [24, 24, 278, 556, 1112, 1280],
        # 3.0: [48, 48, 384, 768, 1536, 2048]
    }

    out_channel_map = sc_ch_dict[s]

    if s > 1:
        drop_p = 0.2
    else:
        drop_p = 0.1

    return CNNModel(
        out_channel_map=out_channel_map,
        rep_layers=rep_layers,
        drop_p=drop_p)


def oth_dicenet_wd5(pretrained=False, **kwargs):
    return get_dicenet(s=0.2)


def oth_dicenet_wd2(pretrained=False, **kwargs):
    return get_dicenet(s=0.5)


def oth_dicenet_w3d4(pretrained=False, **kwargs):
    return get_dicenet(s=0.75)


def oth_dicenet_w1(pretrained=False, **kwargs):
    return get_dicenet(s=1.0)


def oth_dicenet_w3d2(pretrained=False, **kwargs):
    return get_dicenet(s=1.5)


def oth_dicenet_w5d4(pretrained=False, **kwargs):
    return get_dicenet(s=1.25)


def oth_dicenet_w7d8(pretrained=False, **kwargs):
    return get_dicenet(s=1.75)


def oth_dicenet_w2(pretrained=False, **kwargs):
    return get_dicenet(s=2.0)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False

    models = [
        oth_dicenet_wd5,
        oth_dicenet_wd2,
        oth_dicenet_w3d4,
        oth_dicenet_w1,
        oth_dicenet_w3d2,
        oth_dicenet_w5d4,
        oth_dicenet_w7d8,
        oth_dicenet_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_dicenet_wd5 or weight_count == 1130704)
        assert (model != oth_dicenet_wd2 or weight_count == 1214120)
        assert (model != oth_dicenet_w3d4 or weight_count == 1495676)
        assert (model != oth_dicenet_w1 or weight_count == 1805604)
        assert (model != oth_dicenet_w3d2 or weight_count == 2652200)
        assert (model != oth_dicenet_w5d4 or weight_count == 2162888)
        assert (model != oth_dicenet_w7d8 or weight_count == 3264932)
        assert (model != oth_dicenet_w2 or weight_count == 3979044)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
