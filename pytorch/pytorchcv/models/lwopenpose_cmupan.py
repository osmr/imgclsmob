"""
    Lightweight OpenPose 2D/3D for CMU Panoptic, implemented in PyTorch.
    Original paper: 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.
"""

__all__ = ['LwOpenPose', 'lwopenpose2d_mobilenet_cmupan_coco', 'lwopenpose3d_mobilenet_cmupan_coco',
           'LwopDecoderFinalBlock']

import os
import torch
from torch import nn
from .common import conv1x1, conv1x1_block, conv3x3_block, dwsconv3x3_block


class LwopResBottleneck(nn.Module):
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
        super(LwopResBottleneck, self).__init__()
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


class LwopResUnit(nn.Module):
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
        super(LwopResUnit, self).__init__()
        self.activate = activate
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = LwopResBottleneck(
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


class LwopEncoderFinalBlock(nn.Module):
    """
    Lightweight OpenPose 2D/3D specific encoder final block.

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
        super(LwopEncoderFinalBlock, self).__init__()
        self.pre_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)
        self.body = nn.Sequential()
        for i in range(3):
            self.body.add_module("block{}".format(i + 1), dwsconv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                dw_use_bn=False,
                pw_use_bn=False,
                dw_activation=(lambda: nn.ELU(inplace=True)),
                pw_activation=(lambda: nn.ELU(inplace=True))))
        self.post_conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)

    def forward(self, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        x = self.post_conv(x)
        return x


class LwopRefinementBlock(nn.Module):
    """
    Lightweight OpenPose 2D/3D specific refinement block for decoder units.

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
        super(LwopRefinementBlock, self).__init__()
        self.pre_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)
        self.body = nn.Sequential()
        self.body.add_module("block1", conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=True))
        self.body.add_module("block2", conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=2,
            dilation=2,
            bias=True))

    def forward(self, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        return x


class LwopDecoderBend(nn.Module):
    """
    Lightweight OpenPose 2D/3D specific decoder bend block.

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
        super(LwopDecoderBend, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            use_bn=False)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LwopDecoderInitBlock(nn.Module):
    """
    Lightweight OpenPose 2D/3D specific decoder init block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    """
    def __init__(self,
                 in_channels,
                 keypoints):
        super(LwopDecoderInitBlock, self).__init__()
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        bend_mid_channels = 512

        self.body = nn.Sequential()
        for i in range(3):
            self.body.add_module("block{}".format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=True,
                use_bn=False))
        self.heatmap_bend = LwopDecoderBend(
            in_channels=in_channels,
            mid_channels=bend_mid_channels,
            out_channels=num_heatmap)
        self.paf_bend = LwopDecoderBend(
            in_channels=in_channels,
            mid_channels=bend_mid_channels,
            out_channels=num_paf)

    def forward(self, x):
        y = self.body(x)
        heatmap = self.heatmap_bend(y)
        paf = self.paf_bend(y)
        y = torch.cat((x, heatmap, paf), dim=1)
        return y


class LwopDecoderUnit(nn.Module):
    """
    Lightweight OpenPose 2D/3D specific decoder init.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    """
    def __init__(self,
                 in_channels,
                 keypoints):
        super(LwopDecoderUnit, self).__init__()
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        self.features_channels = in_channels - num_heatmap - num_paf

        self.body = nn.Sequential()
        for i in range(5):
            self.body.add_module("block{}".format(i + 1), LwopRefinementBlock(
                in_channels=in_channels,
                out_channels=self.features_channels))
            in_channels = self.features_channels
        self.heatmap_bend = LwopDecoderBend(
            in_channels=self.features_channels,
            mid_channels=self.features_channels,
            out_channels=num_heatmap)
        self.paf_bend = LwopDecoderBend(
            in_channels=self.features_channels,
            mid_channels=self.features_channels,
            out_channels=num_paf)

    def forward(self, x):
        features = x[:, :self.features_channels]
        y = self.body(x)
        heatmap = self.heatmap_bend(y)
        paf = self.paf_bend(y)
        y = torch.cat((features, heatmap, paf), dim=1)
        return y


class LwopDecoderFeaturesBend(nn.Module):
    """
    Lightweight OpenPose 2D/3D specific decoder 3D features bend.

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
        super(LwopDecoderFeaturesBend, self).__init__()
        self.body = nn.Sequential()
        for i in range(2):
            self.body.add_module("block{}".format(i + 1), LwopRefinementBlock(
                in_channels=in_channels,
                out_channels=mid_channels))
            in_channels = mid_channels
        self.features_bend = LwopDecoderBend(
            in_channels=mid_channels,
            mid_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.body(x)
        x = self.features_bend(x)
        return x


class LwopDecoderFinalBlock(nn.Module):
    """
    Lightweight OpenPose 2D/3D specific decoder final block for calcualation 3D poses.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    bottleneck_factor : int
        Bottleneck factor.
    calc_3d_features : bool
        Whether to calculate 3D features.
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 bottleneck_factor,
                 calc_3d_features):
        super(LwopDecoderFinalBlock, self).__init__()
        self.num_heatmap_paf = 3 * keypoints
        self.calc_3d_features = calc_3d_features
        features_out_channels = self.num_heatmap_paf
        features_in_channels = in_channels - features_out_channels

        if self.calc_3d_features:
            self.body = nn.Sequential()
            for i in range(5):
                self.body.add_module("block{}".format(i + 1), LwopResUnit(
                    in_channels=in_channels,
                    out_channels=features_in_channels,
                    bottleneck_factor=bottleneck_factor))
                in_channels = features_in_channels
            self.features_bend = LwopDecoderFeaturesBend(
                in_channels=features_in_channels,
                mid_channels=features_in_channels,
                out_channels=features_out_channels)

    def forward(self, x):
        heatmap_paf_2d = x[:, -self.num_heatmap_paf:]
        if not self.calc_3d_features:
            return heatmap_paf_2d
        x = self.body(x)
        x = self.features_bend(x)
        y = torch.cat((heatmap_paf_2d, x), dim=1)
        return y


class LwOpenPose(nn.Module):
    """
    Lightweight OpenPose 2D/3D model from 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    encoder_channels : list of list of int
        Number of output channels for each encoder unit.
    encoder_paddings : list of list of int
        Padding/dilation value for each encoder unit.
    encoder_init_block_channels : int
        Number of output channels for the encoder initial unit.
    encoder_final_block_channels : int
        Number of output channels for the encoder final unit.
    refinement_units : int
        Number of refinement blocks in the decoder.
    calc_3d_features : bool
        Whether to calculate 3D features.
    return_heatmap : bool, default True
        Whether to return only heatmap.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 192)
        Spatial size of the expected input image.
    keypoints : int, default 19
        Number of keypoints.
    """
    def __init__(self,
                 encoder_channels,
                 encoder_paddings,
                 encoder_init_block_channels,
                 encoder_final_block_channels,
                 refinement_units,
                 calc_3d_features,
                 return_heatmap=True,
                 in_channels=3,
                 in_size=(368, 368),
                 keypoints=19):
        super(LwOpenPose, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap
        self.calc_3d_features = calc_3d_features
        num_heatmap_paf = 3 * keypoints

        self.encoder = nn.Sequential()
        backbone = nn.Sequential()
        backbone.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=encoder_init_block_channels,
            stride=2))
        in_channels = encoder_init_block_channels
        for i, channels_per_stage in enumerate(encoder_channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                padding = encoder_paddings[i][j]
                stage.add_module("unit{}".format(j + 1), dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    dilation=padding))
                in_channels = out_channels
            backbone.add_module("stage{}".format(i + 1), stage)
        self.encoder.add_module("backbone", backbone)
        self.encoder.add_module("final_block", LwopEncoderFinalBlock(
            in_channels=in_channels,
            out_channels=encoder_final_block_channels))
        in_channels = encoder_final_block_channels

        self.decoder = nn.Sequential()
        self.decoder.add_module("init_block", LwopDecoderInitBlock(
            in_channels=in_channels,
            keypoints=keypoints))
        in_channels = encoder_final_block_channels + num_heatmap_paf
        for i in range(refinement_units):
            self.decoder.add_module("unit{}".format(i + 1), LwopDecoderUnit(
                in_channels=in_channels,
                keypoints=keypoints))
        self.decoder.add_module("final_block", LwopDecoderFinalBlock(
            in_channels=in_channels,
            keypoints=keypoints,
            bottleneck_factor=2,
            calc_3d_features=calc_3d_features))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.return_heatmap:
            return x
        else:
            return x


def get_lwopenpose(calc_3d_features,
                   keypoints,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".torch", "models"),
                   **kwargs):
    """
    Create Lightweight OpenPose 2D/3D model with specific parameters.

    Parameters:
    ----------
    calc_3d_features : bool, default False
        Whether to calculate 3D features.
    keypoints : int
        Number of keypoints.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    encoder_channels = [[64], [128, 128], [256, 256, 512, 512, 512, 512, 512, 512]]
    encoder_paddings = [[1], [1, 1], [1, 1, 1, 2, 1, 1, 1, 1]]
    encoder_init_block_channels = 32
    encoder_final_block_channels = 128
    refinement_units = 1

    net = LwOpenPose(
        encoder_channels=encoder_channels,
        encoder_paddings=encoder_paddings,
        encoder_init_block_channels=encoder_init_block_channels,
        encoder_final_block_channels=encoder_final_block_channels,
        refinement_units=refinement_units,
        calc_3d_features=calc_3d_features,
        keypoints=keypoints,
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


def lwopenpose2d_mobilenet_cmupan_coco(keypoints=19, **kwargs):
    """
    Lightweight OpenPose 2D model on the base of MobileNet for CMU Panoptic from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    keypoints : int, default 19
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose(calc_3d_features=False, keypoints=keypoints, model_name="lwopenpose2d_mobilenet_cmupan_coco",
                          **kwargs)


def lwopenpose3d_mobilenet_cmupan_coco(keypoints=19, **kwargs):
    """
    Lightweight OpenPose 3D model on the base of MobileNet for CMU Panoptic from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    keypoints : int, default 19
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose(calc_3d_features=True, keypoints=keypoints, model_name="lwopenpose3d_mobilenet_cmupan_coco",
                          **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    in_size = (368, 368)
    keypoints = 19
    return_heatmap = True
    pretrained = False

    models = [
        (lwopenpose2d_mobilenet_cmupan_coco, "2d"),
        (lwopenpose3d_mobilenet_cmupan_coco, "3d"),
    ]

    for model, model_dim in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lwopenpose2d_mobilenet_cmupan_coco or weight_count == 4091698)
        assert (model != lwopenpose3d_mobilenet_cmupan_coco or weight_count == 5085983)

        batch = 1
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        if model_dim == "2d":
            assert (tuple(y.size()) == (batch, 3 * keypoints, in_size[0] // 8, in_size[0] // 8))
        else:
            assert (tuple(y.size()) == (batch, 6 * keypoints, in_size[0] // 8, in_size[0] // 8))


if __name__ == "__main__":
    _test()
