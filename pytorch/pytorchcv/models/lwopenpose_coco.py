"""
    Lightweight OpenPose for COCO Keypoint, implemented in PyTorch.
    Original paper: 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.
"""

__all__ = ['LwOpenPose', 'lwopenpose_mobilenet_coco']

import os
import torch
from torch import nn
from .common import conv1x1, conv1x1_block, conv3x3_block, dwsconv3x3_block


class ResBottleneck(nn.Module):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False):
        super(ResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor if squeeze_out else in_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            bias=bias)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResUnit(nn.Module):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    activate : bool, default False
        Whether to activate the sum.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False,
                 activate=False):
        super(ResUnit, self).__init__()
        self.activate = activate
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            bottleneck_factor=bottleneck_factor,
            squeeze_out=squeeze_out)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                activation=None)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        if self.activate:
            x = self.activ(x)
        return x


class Cpm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(Cpm, self).__init__()
        self.align = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)
        self.trunk = nn.Sequential()
        for i in range(3):
            self.trunk.add_module("block{}".format(i + 1), dwsconv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bn=False,
                dw_activation=(lambda: nn.ELU(inplace=True)),
                pw_activation=(lambda: nn.ELU(inplace=True))))
        self.conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)

    def forward(self, x):
        x = self.align(x)
        x = x + self.trunk(x)
        x = self.conv(x)
        return x


class InitialStage(nn.Module):
    def __init__(self,
                 num_channels,
                 num_heatmaps,
                 num_pafs):
        super(InitialStage, self).__init__()
        self.trunk = nn.Sequential()
        for i in range(3):
            self.trunk.add_module("block{}".format(i + 1), conv3x3_block(
                in_channels=num_channels,
                out_channels=num_channels,
                bias=True,
                use_bn=False))
        self.heatmaps = nn.Sequential(
            conv1x1_block(
                in_channels=num_channels,
                out_channels=512,
                bias=True,
                use_bn=False),
            conv1x1(
                in_channels=512,
                out_channels=num_heatmaps,
                bias=True),
        )
        self.pafs = nn.Sequential(
            conv1x1_block(
                in_channels=num_channels,
                out_channels=512,
                bias=True,
                use_bn=False),
            conv1x1(
                in_channels=512,
                out_channels=num_pafs,
                bias=True),
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return heatmaps, pafs


class RefinementStageBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(RefinementStageBlock, self).__init__()
        self.initial = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)
        self.trunk = nn.Sequential(
            conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bias=True),
            conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                padding=2,
                dilation=2,
                bias=True),
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heatmaps,
                 num_pafs):
        super(RefinementStage, self).__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bias=True,
                use_bn=False),
            conv1x1(
                in_channels=out_channels,
                out_channels=num_heatmaps,
                bias=True),
        )
        self.pafs = nn.Sequential(
            conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bias=True,
                use_bn=False),
            conv1x1(
                in_channels=out_channels,
                out_channels=num_pafs,
                bias=True),
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageLight(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(RefinementStageLight, self).__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, mid_channels),
            RefinementStageBlock(mid_channels, mid_channels)
        )
        self.feature_maps = nn.Sequential(
            conv1x1_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bias=True,
                use_bn=False),
            conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                bias=True),
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        feature_maps = self.feature_maps(trunk_features)
        return [feature_maps]


class Pose3D(nn.Module):
    def __init__(self,
                 in_channels,
                 num_2d_heatmaps,
                 ratio=2,
                 out_channels=57):
        super(Pose3D, self).__init__()
        self.stem = nn.Sequential(
            ResUnit(
                in_channels=(in_channels + num_2d_heatmaps),
                out_channels=in_channels,
                bottleneck_factor=ratio),
            ResUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                bottleneck_factor=ratio),
            ResUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                bottleneck_factor=ratio),
            ResUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                bottleneck_factor=ratio),
            ResUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                bottleneck_factor=ratio),
        )

        self.prediction = RefinementStageLight(in_channels, in_channels, out_channels)

    def forward(self, x, feature_maps_2d):
        stem = self.stem(torch.cat([x, feature_maps_2d], 1))
        feature_maps = self.prediction(stem)
        return feature_maps


class LwOpenPose(nn.Module):
    def __init__(self,
                 channels,
                 paddings,
                 init_block_channels,
                 num_refinement_stages=1,
                 num_channels=128,
                 num_heatmaps=19,
                 num_pafs=38,
                 in_channels=3,
                 in_size=(256, 256),
                 keypoints=17):
        super(LwOpenPose, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints

        self.model = nn.Sequential()
        self.model.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                padding = paddings[i][j]
                stage.add_module("unit{}".format(j + 1), dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    dilation=padding))
                in_channels = out_channels
            self.model.add_module("stage{}".format(i + 1), stage)

        self.cpm = Cpm(
            in_channels=in_channels,
            out_channels=num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()

        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))
        self.Pose3D = Pose3D(128, num_2d_heatmaps=57)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        model_features = self.model(x)
        backbone_features = self.cpm(model_features)

        stages_output = list(self.initial_stage(backbone_features))
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))
        keypoints2d_maps = stages_output[-2]
        paf_maps = stages_output[-1]
        out = self.Pose3D(backbone_features, torch.cat([stages_output[-2], stages_output[-1]], dim=1))

        return out, keypoints2d_maps, paf_maps


def get_simplepose(model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".torch", "models"),
                   **kwargs):
    """
    Create Lightweight OpenPose model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [[64], [128, 128], [256, 256, 512, 512, 512, 512, 512, 512]]
    paddings = [[1], [1, 1], [1, 1, 1, 2, 1, 1, 1, 1]]
    init_block_channels = 32

    net = LwOpenPose(
        channels=channels,
        paddings=paddings,
        init_block_channels=init_block_channels,
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


def lwopenpose_mobilenet_coco(**kwargs):
    """
    Lightweight OpenPose model on the base of MobileNet for COCO Keypoint from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_simplepose(model_name="lwopenpose_mobilenet_coco", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    in_size = (256, 256)
    pretrained = False

    models = [
        lwopenpose_mobilenet_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lwopenpose_mobilenet_coco or weight_count == 5085983)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y[0][0].size()) == (batch, 57, in_size[0] // 8, in_size[0] // 8))
        assert (tuple(y[1][0].size()) == (19, in_size[0] // 8, in_size[0] // 8))
        assert (tuple(y[2][0].size()) == (38, in_size[0] // 8, in_size[0] // 8))


if __name__ == "__main__":
    _test()
