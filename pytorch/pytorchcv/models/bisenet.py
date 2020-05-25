"""
    BiSeNet for CelebAMask-HQ, implemented in PyTorch.
    Original paper: 'BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1808.00897.
"""

__all__ = ['BiSeNet', 'bisenet_resnet18_celebamaskhq']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, InterpolationBlock, MultiOutputSequential
from .resnet import resnet18


class PyramidPoolingZeroBranch(nn.Module):
    """
    Pyramid pooling zero branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of 2 int
        Spatial size of output image for the upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size):
        super(PyramidPoolingZeroBranch, self).__init__()
        self.in_size = in_size

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.up = InterpolationBlock(
            scale_factor=None,
            mode="nearest",
            align_corners=None)

    def forward(self, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        x = self.up(x, size=in_size)
        return x


class AttentionRefinementBlock(nn.Module):
    """
    Attention refinement block.

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
        super(AttentionRefinementBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=(lambda: nn.Sigmoid()))

    def forward(self, x):
        x = self.conv1(x)
        w = self.pool(x)
        w = self.conv2(w)
        x = x * w
        return x


class PyramidPoolingMainBranch(nn.Module):
    """
    Pyramid pooling main branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    scale_factor : float
        Multiplier for spatial size.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor):
        super(PyramidPoolingMainBranch, self).__init__()
        self.att = AttentionRefinementBlock(
            in_channels=in_channels,
            out_channels=out_channels)
        self.up = InterpolationBlock(
            scale_factor=scale_factor,
            mode="nearest",
            align_corners=None)
        self.conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels)

    def forward(self, x, y):
        x = self.att(x)
        x = x + y
        x = self.up(x)
        x = self.conv(x)
        return x


class FeatureFusion(nn.Module):
    """
    Feature fusion block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    reduction : int, default 4
        Squeeze reduction value.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=4):
        super(FeatureFusion, self).__init__()
        mid_channels = out_channels // reduction

        self.conv_merge = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = conv1x1(
            in_channels=out_channels,
            out_channels=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.conv_merge(x)
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x_att = x * w
        x = x + x_att
        return x


class PyramidPooling(nn.Module):
    """
    Pyramid Pooling module.

    Parameters:
    ----------
    x16_in_channels : int
        Number of input channels for x16.
    x32_in_channels : int
        Number of input channels for x32.
    y_out_channels : int
        Number of output channels for y-outputs.
    y32_out_size : tuple of 2 int
        Spatial size of the y32 tensor.
    """
    def __init__(self,
                 x16_in_channels,
                 x32_in_channels,
                 y_out_channels,
                 y32_out_size):
        super(PyramidPooling, self).__init__()
        z_out_channels = 2 * y_out_channels

        self.pool32 = PyramidPoolingZeroBranch(
            in_channels=x32_in_channels,
            out_channels=y_out_channels,
            in_size=y32_out_size)
        self.pool16 = PyramidPoolingMainBranch(
            in_channels=x32_in_channels,
            out_channels=y_out_channels,
            scale_factor=2)
        self.pool8 = PyramidPoolingMainBranch(
            in_channels=x16_in_channels,
            out_channels=y_out_channels,
            scale_factor=2)
        self.fusion = FeatureFusion(
            in_channels=z_out_channels,
            out_channels=z_out_channels)

    def forward(self, x8, x16, x32):
        y32 = self.pool32(x32)
        y16 = self.pool16(x32, y32)
        y8 = self.pool8(x16, y16)
        z8 = self.fusion(x8, y8)
        return z8, y8, y16


class BiSeHead(nn.Module):
    """
    BiSeNet head (final) block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels):
        super(BiSeHead, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BiSeNet(nn.Module):
    """
    BiSeNet model from 'BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1808.00897.

    Parameters:
    ----------
    backbone : func -> nn.Sequential
        Feature extractor.
    aux : bool, default True
        Whether to output an auxiliary results.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (640, 480)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 backbone,
                 aux=True,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(640, 480),
                 num_classes=19):
        super(BiSeNet, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.num_classes = num_classes
        self.aux = aux
        self.fixed_size = fixed_size

        self.backbone, backbone_out_channels = backbone()

        y_out_channels = backbone_out_channels[0]
        z_out_channels = 2 * y_out_channels
        y32_out_size = (self.in_size[0] // 32, self.in_size[1] // 32) if fixed_size else None
        self.pool = PyramidPooling(
            x16_in_channels=backbone_out_channels[1],
            x32_in_channels=backbone_out_channels[2],
            y_out_channels=y_out_channels,
            y32_out_size=y32_out_size)
        self.head_z8 = BiSeHead(
            in_channels=z_out_channels,
            mid_channels=z_out_channels,
            out_channels=num_classes)
        self.up8 = InterpolationBlock(scale_factor=(8 if fixed_size else None))

        if self.aux:
            mid_channels = y_out_channels // 2
            self.head_y8 = BiSeHead(
                in_channels=y_out_channels,
                mid_channels=mid_channels,
                out_channels=num_classes)
            self.head_y16 = BiSeHead(
                in_channels=y_out_channels,
                mid_channels=mid_channels,
                out_channels=num_classes)
            self.up16 = InterpolationBlock(scale_factor=(16 if fixed_size else None))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        assert (x.shape[2] % 32 == 0) and (x.shape[3] % 32 == 0)

        x8, x16, x32 = self.backbone(x)
        z8, y8, y16 = self.pool(x8, x16, x32)

        z8 = self.head_z8(z8)
        z8 = self.up8(z8)

        if self.aux:
            y8 = self.head_y8(y8)
            y16 = self.head_y16(y16)
            y8 = self.up8(y8)
            y16 = self.up16(y16)
            return z8, y8, y16
        else:
            return z8


def get_bisenet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create BiSeNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = BiSeNet(
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def bisenet_resnet18_celebamaskhq(pretrained_backbone=False, num_classes=19, **kwargs):
    """
    BiSeNet model on the base of ResNet-18 for face segmentation on CelebAMask-HQ from 'BiSeNet: Bilateral Segmentation
    Network for Real-time Semantic Segmentation,' https://arxiv.org/abs/1808.00897.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 19
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    def backbone():
        features_raw = resnet18(pretrained=pretrained_backbone).features
        del features_raw[-1]
        features = MultiOutputSequential(return_last=False)
        features.add_module("init_block", features_raw[0])
        for i, stage in enumerate(features_raw[1:]):
            if i != 0:
                stage.do_output = True
            features.add_module("stage{}".format(i + 1), stage)
        out_channels = [128, 256, 512]
        return features, out_channels
    return get_bisenet(backbone=backbone, num_classes=num_classes, model_name="bisenet_resnet18_celebamaskhq", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    in_size = (640, 480)
    aux = True
    pretrained = False

    models = [
        bisenet_resnet18_celebamaskhq,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != bisenet_resnet18_celebamaskhq or weight_count == 13300416)
        else:
            assert (model != bisenet_resnet18_celebamaskhq or weight_count == 13150272)

        batch = 1
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        ys = net(x)
        y = ys[0] if aux else ys
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
