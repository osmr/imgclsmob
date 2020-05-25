
__all__ = ['oth_bisenet']

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels,
            out_channels,
            stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = self.create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = self.create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = self.create_layer_basic(256, 512, bnum=2, stride=2)

    @staticmethod
    def create_layer_basic(in_chan, out_chan, bnum, stride=1):
        layers = [BasicBlock(in_chan, out_chan, stride=stride)]
        for i in range(bnum - 1):
            layers.append(BasicBlock(out_chan, out_chan, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat8, feat16, feat32


class AttentionRefinementModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        w = self.pool(x)
        w = self.conv_atten(w)
        w = self.bn_atten(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = self.pool(feat32)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16


class FeatureFusionModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(FeatureFusionModule, self).__init__()

        self.convblk = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            out_channels,
            out_channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(
            out_channels // 4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = self.pool(feat)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNetOutput(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv_out = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class BiSeNet(nn.Module):

    def __init__(self,
                 num_classes=19):
        super(BiSeNet, self).__init__()
        self.num_classes = num_classes

        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, num_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, num_classes)
        
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        H, W = x.size()[2:]

        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, size=(H, W), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, size=(H, W), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, size=(H, W), mode="bilinear", align_corners=True)
        return feat_out, feat_out16, feat_out32


def oth_bisenet(num_classes=19, pretrained=False, **kwargs):
    net = BiSeNet(num_classes=num_classes)
    return net


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import numpy as np
    import torch

    pretrained = False

    models = [
        oth_bisenet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_bisenet or weight_count == 13300416)


if __name__ == "__main__":
    _test()
