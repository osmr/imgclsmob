"""
    DANet for image segmentation, implemented in Gluon.
    Original paper: 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
"""

__all__ = ['DANet', 'danet_resnetd50b_cityscapes', 'danet_resnetd101b_cityscapes', 'ScaleBlock']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .common import conv1x1, conv3x3_block
from .resnetd import resnetd50b, resnetd101b


class ScaleBlock(nn.Module):
    """
    Simple scale block.
    """
    def __init__(self):
        super(ScaleBlock, self).__init__()
        self.alpha = Parameter(torch.Tensor((1,)))

    def forward(self, x):
        return self.alpha * x

    def __repr__(self):
        s = '{name}(alpha={alpha})'
        return s.format(
            name=self.__class__.__name__,
            gamma=self.alpha.shape[0])

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        num_flops = x.numel()
        num_macs = 0
        return num_flops, num_macs


class PosAttBlock(nn.Module):
    """
    Position attention block from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
    It captures long-range spatial contextual information.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 8
        Squeeze reduction value.
    """
    def __init__(self,
                 channels,
                 reduction=8):
        super(PosAttBlock, self).__init__()
        mid_channels = channels // reduction

        self.query_conv = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            bias=True)
        self.key_conv = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            bias=True)
        self.value_conv = conv1x1(
            in_channels=channels,
            out_channels=channels,
            bias=True)
        self.scale = ScaleBlock()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        proj_query = self.query_conv(x).view((batch, -1, height * width))
        proj_key = self.key_conv(x).view((batch, -1, height * width))
        proj_value = self.value_conv(x).view((batch, -1, height * width))

        energy = proj_query.transpose(1, 2).contiguous().bmm(proj_key)
        w = self.softmax(energy)

        y = proj_value.bmm(w.transpose(1, 2).contiguous())
        y = y.reshape((batch, -1, height, width))
        y = self.scale(y) + x
        return y


class ChaAttBlock(nn.Module):
    """
    Channel attention block from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
    It explicitly models interdependencies between channels.
    """
    def __init__(self):
        super(ChaAttBlock, self).__init__()
        self.scale = ScaleBlock()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        proj_query = x.view((batch, -1, height * width))
        proj_key = x.view((batch, -1, height * width))
        proj_value = x.view((batch, -1, height * width))

        energy = proj_query.bmm(proj_key.transpose(1, 2).contiguous())
        energy_max, _ = energy.max(dim=-1, keepdims=True)
        energy_new = energy_max.expand_as(energy) - energy
        w = self.softmax(energy_new)

        y = w.bmm(proj_value)
        y = y.reshape((batch, -1, height, width))
        y = self.scale(y) + x
        return y


class DANetHeadBranch(nn.Module):
    """
    DANet head branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    pose_att : bool, default True
        Whether to use position attention instead of channel one.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pose_att=True):
        super(DANetHeadBranch, self).__init__()
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        if pose_att:
            self.att = PosAttBlock(mid_channels)
        else:
            self.att = ChaAttBlock()
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.att(x)
        y = self.conv2(x)
        x = self.conv3(y)
        x = self.dropout(x)
        return x, y


class DANetHead(nn.Module):
    """
    DANet head block.

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
        super(DANetHead, self).__init__()
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        self.branch_pa = DANetHeadBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            pose_att=True)
        self.branch_ca = DANetHeadBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            pose_att=False)
        self.conv = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

    def forward(self, x):
        pa_x, pa_y = self.branch_pa(x)
        ca_x, ca_y = self.branch_ca(x)
        y = pa_y + ca_y
        x = self.conv(y)
        x = self.dropout(x)
        return x, pa_x, ca_x


class DANet(nn.Module):
    """
    DANet model from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int, default 2048
        Number of output channels form feature extractor.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    num_classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 num_classes=19):
        super(DANet, self).__init__()
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.aux = aux
        self.fixed_size = fixed_size

        self.backbone = backbone
        self.head = DANetHead(
            in_channels=backbone_out_channels,
            out_channels=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x, _ = self.backbone(x)
        x, y, z = self.head(x)
        x = F.interpolate(x, size=in_size, mode="bilinear", align_corners=True)
        if self.aux:
            y = F.interpolate(y, size=in_size, mode="bilinear", align_corners=True)
            z = F.interpolate(z, size=in_size, mode="bilinear", align_corners=True)
            return x, y, z
        else:
            return x


def get_danet(backbone,
              num_classes,
              aux=False,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create DANet model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    num_classes : int
        Number of segmentation classes.
    aux : bool, default False
        Whether to output an auxiliary result.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = DANet(
        backbone=backbone,
        num_classes=num_classes,
        aux=aux,
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


def danet_resnetd50b_cityscapes(pretrained_backbone=False, num_classes=19, aux=True, **kwargs):
    """
    DANet model on the base of ResNet(D)-50b for Cityscapes from 'Dual Attention Network for Scene Segmentation,'
    https://arxiv.org/abs/1809.02983.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone[-1]
    return get_danet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="danet_resnetd50b_cityscapes",
                     **kwargs)


def danet_resnetd101b_cityscapes(pretrained_backbone=False, num_classes=19, aux=True, **kwargs):
    """
    DANet model on the base of ResNet(D)-101b for Cityscapes from 'Dual Attention Network for Scene Segmentation,'
    https://arxiv.org/abs/1809.02983.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone[-1]
    return get_danet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="danet_resnetd101b_cityscapes",
                     **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    in_size = (480, 480)
    aux = True
    pretrained = False

    models = [
        danet_resnetd50b_cityscapes,
        danet_resnetd101b_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != danet_resnetd50b_cityscapes or weight_count == 47586427)
        assert (model != danet_resnetd101b_cityscapes or weight_count == 66578555)

        batch = 2
        num_classes = 19
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        ys = net(x)
        y = ys[0] if aux else ys
        y.sum().backward()
        assert ((y.size(0) == x.size(0)) and (y.size(1) == num_classes) and (y.size(2) == x.size(2)) and
                (y.size(3) == x.size(3)))


if __name__ == "__main__":
    _test()
