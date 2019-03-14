"""
    PSPNet, implemented in PyTorch.
    Original paper: 'Pyramid Scene Parsing Network,' https://arxiv.org/abs/1612.01105.
"""

__all__ = ['PSPNet', 'pspnet_resnet50_voc', 'pspnet_resnet101_voc', 'pspnet_resnet50_ade20k']

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent, Identity
from .resnet import resnet50, resnet101


class PyramidPoolingBranch(nn.Module):
    """
    Pyramid Pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    output_size : int
        Target output size of the image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 output_size):
        super(PyramidPoolingBranch, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class PyramidPooling(nn.Module):
    """
    Pyramid Pooling module.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PyramidPooling, self).__init__()
        output_sizes = [1, 2, 3, 6]
        assert (len(output_sizes) == 4)
        assert (in_channels % 4 == 0)
        mid_channels = in_channels // 4

        self.branches = Concurrent()
        self.branches.add_module("branch1", Identity())
        for i, output_size in enumerate(output_sizes):
            self.branches.add_module("branch{}".format(i + 2), PyramidPoolingBranch(
                in_channels=in_channels,
                out_channels=mid_channels,
                output_size=output_size))

    def forward(self, x):
        x = self.branches(x)
        return x


class PSPNet(nn.Module):
    """
    PSPNet model from 'Pyramid Scene Parsing Network,' https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int, default 2048
        Number of output channels form feature extractor.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=21):
        super(PSPNet, self).__init__()
        assert (in_channels > 0)
        self.in_size = in_size
        self.num_classes = num_classes
        pool_out_channels = 2 * backbone_out_channels
        mid_channels = backbone_out_channels // 4

        self.backbone = backbone
        self.pool = PyramidPooling(in_channels=backbone_out_channels)
        self.conv1 = conv3x3_block(
            in_channels=pool_out_channels,
            out_channels=mid_channels)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.backbone(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


def get_pspnet(backbone,
               num_classes,
               model_name=None,
               pretrained=False,
               root=os.path.join('~', '.torch', 'models'),
               **kwargs):
    """
    Create PSPNet model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    num_classes : int, default 21
        Number of segmentation classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    net = PSPNet(
        backbone=backbone,
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


def pspnet_resnet50_voc(pretrained_backbone=False, num_classes=21, **kwargs):
    """
    PSPNet model on the base of ResNet-50 for Pascal VOC from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 21
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50(pretrained=pretrained_backbone).features[:-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, model_name="pspnet_resnet50_voc", **kwargs)


def pspnet_resnet101_voc(pretrained_backbone=False, num_classes=21, **kwargs):
    """
    PSPNet model on the base of ResNet-101 for Pascal VOC from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 21
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101(pretrained=pretrained_backbone).features[:-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, model_name="pspnet_resnet101_voc", **kwargs)


def pspnet_resnet50_ade20k(pretrained_backbone=False, num_classes=150, **kwargs):
    """
    PSPNet model on the base of ResNet-50 for ADE20K from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 150
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50(pretrained=pretrained_backbone).features[:-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, model_name="pspnet_resnet50_ade20k", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        (pspnet_resnet50_voc, 21),
        (pspnet_resnet101_voc, 21),
        (pspnet_resnet50_ade20k, 150),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pspnet_resnet50_voc or weight_count == 46592576)
        assert (model != pspnet_resnet101_voc or weight_count == 65584704)
        assert (model != pspnet_resnet50_ade20k or weight_count == 46658624)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        y.sum().backward()
        assert ((y.size(0) == x.size(0)) and (y.size(1) == num_classes) and (y.size(2) == x.size(2)) and
                (y.size(3) == x.size(3)))


if __name__ == "__main__":
    _test()
