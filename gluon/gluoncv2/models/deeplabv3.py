"""
    DeepLabv3 for image segmentation, implemented in Gluon.
    Original paper: 'Rethinking Atrous Convolution for Semantic Image Segmentation,' https://arxiv.org/abs/1706.05587.
"""

__all__ = ['DeepLabv3', 'deeplabv3_resnetd50b_voc', 'deeplabv3_resnetd101b_voc', 'deeplabv3_resnetd152b_voc',
           'deeplabv3_resnetd50b_coco', 'deeplabv3_resnetd101b_coco', 'deeplabv3_resnetd152b_coco',
           'deeplabv3_resnetd50b_ade20k', 'deeplabv3_resnetd101b_ade20k', 'deeplabv3_resnetd50b_cityscapes',
           'deeplabv3_resnetd101b_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import conv1x1, conv1x1_block, conv3x3_block
from .resnetd import resnetd50b, resnetd101b, resnetd152b


class DeepLabv3FinalBlock(HybridBlock):
    """
    DeepLabv3 final block.

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
                 bottleneck_factor=4,
                 **kwargs):
        super(DeepLabv3FinalBlock, self).__init__(**kwargs)
        assert (in_channels % bottleneck_factor == 0)
        mid_channels = in_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.dropout = nn.Dropout(rate=0.1)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def hybrid_forward(self, F, x, out_size):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.contrib.BilinearResize2D(x, height=out_size[0], width=out_size[1])
        return x


class ASPPAvgBranch(HybridBlock):
    """
    ASPP branch with average pooling.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    upscale_out_size : tuple of 2 int or None
        Spatial size of output image for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 upscale_out_size,
                 **kwargs):
        super(ASPPAvgBranch, self).__init__(**kwargs)
        self.upscale_out_size = upscale_out_size

        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels)

    def hybrid_forward(self, F, x):
        in_size = self.upscale_out_size if self.upscale_out_size is not None else x.shape[2:]
        x = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        x = self.conv(x)
        x = F.contrib.BilinearResize2D(x, height=in_size[0], width=in_size[1])
        return x


class AtrousSpatialPyramidPooling(HybridBlock):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    upscale_out_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 upscale_out_size,
                 **kwargs):
        super(AtrousSpatialPyramidPooling, self).__init__(**kwargs)
        atrous_rates = [12, 24, 36]
        assert (in_channels % 8 == 0)
        mid_channels = in_channels // 8
        project_in_channels = 5 * mid_channels

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels))
            for atrous_rate in atrous_rates:
                self.branches.add(conv3x3_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    padding=atrous_rate,
                    dilation=atrous_rate))
            self.branches.add(ASPPAvgBranch(
                in_channels=in_channels,
                out_channels=mid_channels,
                upscale_out_size=upscale_out_size))
            self.conv = conv1x1_block(
                in_channels=project_in_channels,
                out_channels=mid_channels)
            self.dropout = nn.Dropout(rate=0.5)

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class DeepLabv3(HybridBlock):
    """
    DeepLabv3 model from 'Rethinking Atrous Convolution for Semantic Image Segmentation,'
    https://arxiv.org/abs/1706.05587.

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
    classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 classes=21,
                 **kwargs):
        super(DeepLabv3, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size

        with self.name_scope():
            self.backbone = backbone
            pool_out_size = (self.in_size[0] // 8, self.in_size[1] // 8) if fixed_size else None
            self.pool = AtrousSpatialPyramidPooling(
                in_channels=backbone_out_channels,
                upscale_out_size=pool_out_size)
            pool_out_channels = backbone_out_channels // 8
            self.final_block = DeepLabv3FinalBlock(
                in_channels=pool_out_channels,
                out_channels=classes,
                bottleneck_factor=1)
            if self.aux:
                aux_out_channels = backbone_out_channels // 2
                self.aux_block = DeepLabv3FinalBlock(
                    in_channels=aux_out_channels,
                    out_channels=classes,
                    bottleneck_factor=4)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x, y = self.backbone(x)
        x = self.pool(x)
        x = self.final_block(x, in_size)
        if self.aux:
            y = self.aux_block(y, in_size)
            return x, y
        else:
            return x


def get_deeplabv3(backbone,
                  classes,
                  aux=False,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create DeepLabv3 model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    classes : int
        Number of segmentation classes.
    aux : bool, default False
        Whether to output an auxiliary result.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    net = DeepLabv3(
        backbone=backbone,
        classes=classes,
        aux=aux,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx,
            ignore_extra=True)

    return net


def deeplabv3_resnetd50b_voc(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for Pascal VOC from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_voc", **kwargs)


def deeplabv3_resnetd101b_voc(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for Pascal VOC from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_voc", **kwargs)


def deeplabv3_resnetd152b_voc(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-152b for Pascal VOC from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd152b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd152b_voc", **kwargs)


def deeplabv3_resnetd50b_coco(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for COCO from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_coco", **kwargs)


def deeplabv3_resnetd101b_coco(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for COCO from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_coco", **kwargs)


def deeplabv3_resnetd152b_coco(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-152b for COCO from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd152b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd152b_coco", **kwargs)


def deeplabv3_resnetd50b_ade20k(pretrained_backbone=False, classes=150, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for ADE20K from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 150
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_ade20k",
                         **kwargs)


def deeplabv3_resnetd101b_ade20k(pretrained_backbone=False, classes=150, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for ADE20K from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 150
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_ade20k",
                         **kwargs)


def deeplabv3_resnetd50b_cityscapes(pretrained_backbone=False, classes=19, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for Cityscapes from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_cityscapes",
                         **kwargs)


def deeplabv3_resnetd101b_cityscapes(pretrained_backbone=False, classes=19, aux=True, **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for Cityscapes from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_cityscapes",
                         **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (480, 480)
    aux = False
    pretrained = False

    models = [
        (deeplabv3_resnetd50b_voc, 21),
        (deeplabv3_resnetd101b_voc, 21),
        (deeplabv3_resnetd152b_voc, 21),
        (deeplabv3_resnetd50b_coco, 21),
        (deeplabv3_resnetd101b_coco, 21),
        (deeplabv3_resnetd152b_coco, 21),
        (deeplabv3_resnetd50b_ade20k, 150),
        (deeplabv3_resnetd101b_ade20k, 150),
        (deeplabv3_resnetd50b_cityscapes, 19),
        (deeplabv3_resnetd101b_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != deeplabv3_resnetd50b_voc or weight_count == 42127850)
            assert (model != deeplabv3_resnetd101b_voc or weight_count == 61119978)
            assert (model != deeplabv3_resnetd152b_voc or weight_count == 76763626)
            assert (model != deeplabv3_resnetd50b_coco or weight_count == 42127850)
            assert (model != deeplabv3_resnetd101b_coco or weight_count == 61119978)
            assert (model != deeplabv3_resnetd152b_coco or weight_count == 76763626)
            assert (model != deeplabv3_resnetd50b_ade20k or weight_count == 42194156)
            assert (model != deeplabv3_resnetd101b_ade20k or weight_count == 61186284)
            assert (model != deeplabv3_resnetd50b_cityscapes or weight_count == 42126822)
            assert (model != deeplabv3_resnetd101b_cityscapes or weight_count == 61118950)
        else:
            assert (model != deeplabv3_resnetd50b_voc or weight_count == 39762645)
            assert (model != deeplabv3_resnetd101b_voc or weight_count == 58754773)
            assert (model != deeplabv3_resnetd152b_voc or weight_count == 74398421)
            assert (model != deeplabv3_resnetd50b_coco or weight_count == 39762645)
            assert (model != deeplabv3_resnetd101b_coco or weight_count == 58754773)
            assert (model != deeplabv3_resnetd152b_coco or weight_count == 74398421)
            assert (model != deeplabv3_resnetd50b_ade20k or weight_count == 39795798)
            assert (model != deeplabv3_resnetd101b_ade20k or weight_count == 58787926)
            assert (model != deeplabv3_resnetd50b_cityscapes or weight_count == 39762131)
            assert (model != deeplabv3_resnetd101b_cityscapes or weight_count == 58754259)

        x = mx.nd.zeros((1, 3, in_size[0], in_size[1]), ctx=ctx)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
