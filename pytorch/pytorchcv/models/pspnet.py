"""
    PSPNet for image segmentation, implemented in PyTorch.
    Original paper: 'Pyramid Scene Parsing Network,' https://arxiv.org/abs/1612.01105.
"""

__all__ = ['PSPNet', 'pspnet_resnetd50b_voc', 'pspnet_resnetd101b_voc', 'pspnet_resnetd50b_coco',
           'pspnet_resnetd101b_coco', 'pspnet_resnetd50b_ade20k', 'pspnet_resnetd101b_ade20k',
           'pspnet_resnetd50b_cityscapes', 'pspnet_resnetd101b_cityscapes']

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent, Identity
from .resnetd import resnetd50b, resnetd101b


class PSPFinalBlock(nn.Module):
    """
    PSPNet final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bottleneck_factor=4):
        super(PSPFinalBlock, self).__init__()
        assert (in_channels % bottleneck_factor == 0)
        mid_channels = in_channels // bottleneck_factor

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x, out_size):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=True)
        return x


class PyramidPoolingBranch(nn.Module):
    """
    Pyramid Pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    pool_out_size : int
        Target output size of the image.
    upscale_out_size : tuple of 2 int
        Spatial size of output image for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_out_size,
                 upscale_out_size):
        super(PyramidPoolingBranch, self).__init__()
        self.upscale_out_size = upscale_out_size

        self.pool = nn.AdaptiveAvgPool2d(pool_out_size)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        in_size = self.upscale_out_size if self.upscale_out_size is not None else x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=in_size, mode="bilinear", align_corners=True)
        return x


class PyramidPooling(nn.Module):
    """
    Pyramid Pooling module.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    upscale_out_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 upscale_out_size):
        super(PyramidPooling, self).__init__()
        pool_out_sizes = [1, 2, 3, 6]
        assert (len(pool_out_sizes) == 4)
        assert (in_channels % 4 == 0)
        mid_channels = in_channels // 4

        self.branches = Concurrent()
        self.branches.add_module("branch1", Identity())
        for i, pool_out_size in enumerate(pool_out_sizes):
            self.branches.add_module("branch{}".format(i + 2), PyramidPoolingBranch(
                in_channels=in_channels,
                out_channels=mid_channels,
                pool_out_size=pool_out_size,
                upscale_out_size=upscale_out_size))

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
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    num_classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 num_classes=21):
        super(PSPNet, self).__init__()
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.aux = aux
        self.fixed_size = fixed_size

        self.backbone = backbone
        pool_out_size = (self.in_size[0] // 8, self.in_size[1] // 8) if fixed_size else None
        self.pool = PyramidPooling(
            in_channels=backbone_out_channels,
            upscale_out_size=pool_out_size)
        pool_out_channels = 2 * backbone_out_channels
        self.final_block = PSPFinalBlock(
            in_channels=pool_out_channels,
            out_channels=num_classes,
            bottleneck_factor=8)
        if self.aux:
            aux_out_channels = backbone_out_channels // 2
            self.aux_block = PSPFinalBlock(
                in_channels=aux_out_channels,
                out_channels=num_classes,
                bottleneck_factor=4)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x, y = self.backbone(x)
        x = self.pool(x)
        x = self.final_block(x, in_size)
        if self.aux:
            y = self.aux_block(y, in_size)
            return x, y
        else:
            return x


def get_pspnet(backbone,
               num_classes,
               aux=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create PSPNet model with specific parameters.

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

    net = PSPNet(
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


def pspnet_resnetd50b_voc(pretrained_backbone=False, num_classes=21, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-50b for Pascal VOC from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd50b_voc", **kwargs)


def pspnet_resnetd101b_voc(pretrained_backbone=False, num_classes=21, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-101b for Pascal VOC from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd101b_voc",
                      **kwargs)


def pspnet_resnetd50b_coco(pretrained_backbone=False, num_classes=21, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-50b for COCO from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd50b_coco",
                      **kwargs)


def pspnet_resnetd101b_coco(pretrained_backbone=False, num_classes=21, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-101b for COCO from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd101b_coco",
                      **kwargs)


def pspnet_resnetd50b_ade20k(pretrained_backbone=False, num_classes=150, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-50b for ADE20K from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 150
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd50b_ade20k",
                      **kwargs)


def pspnet_resnetd101b_ade20k(pretrained_backbone=False, num_classes=150, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-101b for ADE20K from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 150
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd101b_ade20k",
                      **kwargs)


def pspnet_resnetd50b_cityscapes(pretrained_backbone=False, num_classes=19, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-50b for Cityscapes from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

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
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd50b_cityscapes",
                      **kwargs)


def pspnet_resnetd101b_cityscapes(pretrained_backbone=False, num_classes=19, aux=True, **kwargs):
    """
    PSPNet model on the base of ResNet(D)-101b for Cityscapes from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

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
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, multi_output=True).features
    del backbone[-1]
    return get_pspnet(backbone=backbone, num_classes=num_classes, aux=aux, model_name="pspnet_resnetd101b_cityscapes",
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
    aux = False
    pretrained = False

    models = [
        (pspnet_resnetd50b_voc, 21),
        (pspnet_resnetd101b_voc, 21),
        (pspnet_resnetd50b_coco, 21),
        (pspnet_resnetd101b_coco, 21),
        (pspnet_resnetd50b_ade20k, 150),
        (pspnet_resnetd101b_ade20k, 150),
        (pspnet_resnetd50b_cityscapes, 19),
        (pspnet_resnetd101b_cityscapes, 19),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != pspnet_resnetd50b_voc or weight_count == 49081578)
            assert (model != pspnet_resnetd101b_voc or weight_count == 68073706)
            assert (model != pspnet_resnetd50b_coco or weight_count == 49081578)
            assert (model != pspnet_resnetd101b_coco or weight_count == 68073706)
            assert (model != pspnet_resnetd50b_ade20k or weight_count == 49180908)
            assert (model != pspnet_resnetd101b_ade20k or weight_count == 68173036)
            assert (model != pspnet_resnetd50b_cityscapes or weight_count == 49080038)
            assert (model != pspnet_resnetd101b_cityscapes or weight_count == 68072166)
        else:
            assert (model != pspnet_resnetd50b_voc or weight_count == 46716373)
            assert (model != pspnet_resnetd101b_voc or weight_count == 65708501)
            assert (model != pspnet_resnetd50b_coco or weight_count == 46716373)
            assert (model != pspnet_resnetd101b_coco or weight_count == 65708501)
            assert (model != pspnet_resnetd50b_ade20k or weight_count == 46782550)
            assert (model != pspnet_resnetd101b_ade20k or weight_count == 65774678)
            assert (model != pspnet_resnetd50b_cityscapes or weight_count == 46715347)
            assert (model != pspnet_resnetd101b_cityscapes or weight_count == 65707475)

        x = torch.randn(1, 3, in_size[0], in_size[1])
        ys = net(x)
        y = ys[0] if aux else ys
        y.sum().backward()
        assert ((y.size(0) == x.size(0)) and (y.size(1) == num_classes) and (y.size(2) == x.size(2)) and
                (y.size(3) == x.size(3)))


if __name__ == "__main__":
    _test()
