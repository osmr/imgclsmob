"""
    CenterNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Objects as Points,' https://arxiv.org/abs/1904.07850.
"""

__all__ = ['CenterNet', 'centernet_resnet18_voc', 'centernet_resnet18_coco', 'centernet_resnet50b_voc',
           'centernet_resnet50b_coco', 'centernet_resnet101b_voc', 'centernet_resnet101b_coco',
           'CenterNetHeatmapMaxDet']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv3x3_block, DeconvBlock, Concurrent
from .resnet import resnet18, resnet50b, resnet101b


class CenterNetDecoderUnit(nn.Module):
    """
    CenterNet decoder unit.

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
        super(CenterNetDecoderUnit, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True)
        self.deconv = DeconvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


class CenterNetHeadBlock(nn.Module):
    """
    CenterNet simple head block.

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
        super(CenterNetHeadBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            bias=True,
            use_bn=False)
        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterNetHeatmapBlock(nn.Module):
    """
    CenterNet heatmap block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    do_nms : bool
        Whether do NMS (or simply clip for training otherwise).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 do_nms):
        super(CenterNetHeatmapBlock, self).__init__()
        self.do_nms = do_nms

        self.head = CenterNetHeadBlock(
            in_channels=in_channels,
            out_channels=out_channels)
        self.sigmoid = nn.Sigmoid()
        if self.do_nms:
            self.pool = nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.sigmoid(x)
        if self.do_nms:
            y = self.pool(x)
            x = x * (y == x)
        else:
            eps = 1e-4
            x = x.clamp(min=eps, max=(1.0 - eps))
        return x


class CenterNetHeatmapMaxDet(nn.Module):
    """
    CenterNet decoder for heads (heatmap, wh, reg).

    Parameters:
    ----------
    topk : int, default 40
        Keep only `topk` detections.
    scale : int, default is 4
        Downsampling scale factor.
    """
    def __init__(self,
                 topk=40,
                 scale=4):
        super(CenterNetHeatmapMaxDet, self).__init__()
        self.topk = topk
        self.scale = scale

    def forward(self, x):
        heatmap = x[:, :-4]
        wh = x[:, -4:-2]
        reg = x[:, -2:]
        batch, _, out_h, out_w = heatmap.shape
        scores, indices = heatmap.view((batch, -1)).topk(k=self.topk)
        topk_classes = (indices / (out_h * out_w)).type(torch.float32)
        topk_indices = indices.fmod(out_h * out_w)
        topk_ys = (topk_indices / out_w).type(torch.float32)
        topk_xs = topk_indices.fmod(out_w).type(torch.float32)
        center = reg.permute(0, 2, 3, 1).view((batch, -1, 2))
        wh = wh.permute(0, 2, 3, 1).view((batch, -1, 2))
        xs = torch.gather(center[:, :, 0], dim=-1, index=topk_indices)
        ys = torch.gather(center[:, :, 1], dim=-1, index=topk_indices)
        topk_xs = topk_xs + xs
        topk_ys = topk_ys + ys
        w = torch.gather(wh[:, :, 0], dim=-1, index=topk_indices)
        h = torch.gather(wh[:, :, 1], dim=-1, index=topk_indices)
        half_w = 0.5 * w
        half_h = 0.5 * h
        bboxes = torch.stack((topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h), dim=-1)

        bboxes = bboxes * self.scale
        topk_classes = topk_classes.unsqueeze(dim=-1)
        scores = scores.unsqueeze(dim=-1)
        result = torch.cat((bboxes, topk_classes, scores), dim=-1)
        return result

    def __repr__(self):
        s = "{name}(topk={topk}, scale={scale})"
        return s.format(
            name=self.__class__.__name__,
            topk=self.topk,
            scale=self.scale)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        num_flops = 10 * x.size
        num_macs = 0
        return num_flops, num_macs


class CenterNet(nn.Module):
    """
    CenterNet model from 'Objects as Points,' https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    channels : list of int
        Number of output channels for each decoder unit.
    return_heatmap : bool, default False
        Whether to return only heatmap.
    topk : int, default 40
        Keep only `topk` detections.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (512, 512)
        Spatial size of the expected input image.
    num_classes : int, default 80
        Number of classification classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels,
                 channels,
                 return_heatmap=False,
                 topk=40,
                 in_channels=3,
                 in_size=(512, 512),
                 num_classes=80):
        super(CenterNet, self).__init__()
        self.in_size = in_size
        self.in_channels = in_channels
        self.return_heatmap = return_heatmap

        self.backbone = backbone

        self.decoder = nn.Sequential()
        in_channels = backbone_out_channels
        for i, out_channels in enumerate(channels):
            self.decoder.add_module("unit{}".format(i + 1), CenterNetDecoderUnit(
                in_channels=in_channels,
                out_channels=out_channels))
            in_channels = out_channels

        heads = Concurrent()
        heads.add_module("heapmap_block", CenterNetHeatmapBlock(
            in_channels=in_channels,
            out_channels=num_classes,
            do_nms=(not self.return_heatmap)))
        heads.add_module("wh_block", CenterNetHeadBlock(
            in_channels=in_channels,
            out_channels=2))
        heads.add_module("reg_block", CenterNetHeadBlock(
            in_channels=in_channels,
            out_channels=2))
        self.decoder.add_module("heads", heads)

        if not self.return_heatmap:
            self.heatmap_max_det = CenterNetHeatmapMaxDet(
                topk=topk,
                scale=4)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        if not self.return_heatmap:
            x = self.heatmap_max_det(x)
        return x


def get_centernet(backbone,
                  backbone_out_channels,
                  num_classes,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create CenterNet model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    num_classes : int
        Number of classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns:
    -------
    nn.Module
        A network.
    """
    channels = [256, 128, 64]

    net = CenterNet(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
        num_classes=num_classes,
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


def centernet_resnet18_voc(pretrained_backbone=False, num_classes=20, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 20
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=512, num_classes=num_classes,
                         model_name="centernet_resnet18_voc", **kwargs)


def centernet_resnet18_coco(pretrained_backbone=False, num_classes=80, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 80
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=512, num_classes=num_classes,
                         model_name="centernet_resnet18_coco", **kwargs)


def centernet_resnet50b_voc(pretrained_backbone=False, num_classes=20, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 20
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, num_classes=num_classes,
                         model_name="centernet_resnet50b_voc", **kwargs)


def centernet_resnet50b_coco(pretrained_backbone=False, num_classes=80, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 80
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, num_classes=num_classes,
                         model_name="centernet_resnet50b_coco", **kwargs)


def centernet_resnet101b_voc(pretrained_backbone=False, num_classes=20, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 20
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, num_classes=num_classes,
                         model_name="centernet_resnet101b_voc", **kwargs)


def centernet_resnet101b_coco(pretrained_backbone=False, num_classes=80, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 80
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, num_classes=num_classes,
                         model_name="centernet_resnet101b_coco", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    in_size = (512, 512)
    topk = 40
    return_heatmap = False
    pretrained = False

    models = [
        (centernet_resnet18_voc, 20),
        (centernet_resnet18_coco, 80),
        (centernet_resnet50b_voc, 20),
        (centernet_resnet50b_coco, 80),
        (centernet_resnet101b_voc, 20),
        (centernet_resnet101b_coco, 80),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, topk=topk, in_size=in_size, return_heatmap=return_heatmap)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != centernet_resnet18_voc or weight_count == 14215640)
        assert (model != centernet_resnet18_coco or weight_count == 14219540)
        assert (model != centernet_resnet50b_voc or weight_count == 30086104)
        assert (model != centernet_resnet50b_coco or weight_count == 30090004)
        assert (model != centernet_resnet101b_voc or weight_count == 49078232)
        assert (model != centernet_resnet101b_coco or weight_count == 49082132)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        assert (y.shape[0] == batch)
        if return_heatmap:
            assert (y.shape[1] == classes + 4) and (y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4)
        else:
            assert (y.shape[1] == topk) and (y.shape[2] == 6)


if __name__ == "__main__":
    _test()
