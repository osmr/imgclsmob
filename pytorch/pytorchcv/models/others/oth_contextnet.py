##################################################################################
#ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time
#Paper-Link: https://arxiv.org/abs/1805.04554
##################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["ContextNet"]


class Custom_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(Custom_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class DepthSepConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthSepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class DepthConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            Custom_Conv(in_channels, in_channels * t, 1),
            DepthConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

    
class Shallow_net(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=64, out_channels=128, **kwargs):
        super(Shallow_net, self).__init__()
        self.conv = Custom_Conv(3, dw_channels1, 3, 2)
        self.dsconv1 = DepthSepConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DepthSepConv(dw_channels2, out_channels, 2)
        self.dsconv3 = DepthSepConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        return x


class Deep_net(nn.Module):
    def __init__(self, in_channels, block_channels,
                 t, num_blocks, **kwargs):
        super(Deep_net, self).__init__()
        self.block_channels = block_channels
        self.t = t
        self.num_blocks = num_blocks

        self.conv_ = Custom_Conv(3, in_channels, 3, 2)
        self.bottleneck1 = self._layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t[0], 1)
        self.bottleneck2 = self._layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t[1], 1)
        self.bottleneck3 = self._layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t[2], 2)
        self.bottleneck4 = self._layer(LinearBottleneck, block_channels[2], block_channels[3], num_blocks[3], t[3], 2)
        self.bottleneck5 = self._layer(LinearBottleneck, block_channels[3], block_channels[4], num_blocks[4], t[4], 1)
        self.bottleneck6 = self._layer(LinearBottleneck, block_channels[4], block_channels[5], num_blocks[5], t[5], 1)

    def _layer(self, block, in_channels, out_channels, blocks, t, stride):
        layers = []
        layers.append(block(in_channels, out_channels, t, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, t, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DepthConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h,w), mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = DepthSepConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DepthSepConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, classes, aux=False, **kwargs):
        super(ContextNet, self).__init__()
        self.aux = aux
        self.spatial_detail = Shallow_net(32, 64, 128)
        self.context_feature_extractor = Deep_net(32, [32, 32, 48, 64, 96, 128], [1, 6, 6, 6, 6, 6], [1, 1, 3, 3, 2, 2])
        self.feature_fusion = FeatureFusionModule(128, 128, 128)
        self.classifier = Classifer(128, classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]

        higher_res_features = self.spatial_detail(x)

        x_low = F.interpolate(x, scale_factor = 0.25, mode='bilinear', align_corners=True)

        x = self.context_feature_extractor(x_low)

        x = self.feature_fusion(higher_res_features, x)

        x = self.classifier(x)

        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return x

        # return tuple(outputs)


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextNet(classes=19).to(device)
    summary(model,(3,512,1024))
