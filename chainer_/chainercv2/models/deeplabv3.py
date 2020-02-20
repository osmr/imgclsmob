"""
    DeepLabv3 for image segmentation, implemented in Chainer.
    Original paper: 'Rethinking Atrous Convolution for Semantic Image Segmentation,' https://arxiv.org/abs/1706.05587.
"""

__all__ = ['DeepLabv3', 'deeplabv3_resnetd50b_voc', 'deeplabv3_resnetd101b_voc', 'deeplabv3_resnetd152b_voc',
           'deeplabv3_resnetd50b_coco', 'deeplabv3_resnetd101b_coco', 'deeplabv3_resnetd152b_coco',
           'deeplabv3_resnetd50b_ade20k', 'deeplabv3_resnetd101b_ade20k', 'deeplabv3_resnetd50b_cityscapes',
           'deeplabv3_resnetd101b_cityscapes']

import os
import chainer.functions as F
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent
from .resnetd import resnetd50b, resnetd101b, resnetd152b


class DeepLabv3FinalBlock(Chain):
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
                 bottleneck_factor=4):
        super(DeepLabv3FinalBlock, self).__init__()
        assert (in_channels % bottleneck_factor == 0)
        mid_channels = in_channels // bottleneck_factor

        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.dropout = partial(
                F.dropout,
                ratio=0.1)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def __call__(self, x, out_size):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.resize_images(x, output_shape=out_size)
        return x


class ASPPAvgBranch(Chain):
    """
    ASPP branch with average pooling.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    upscale_out_size : tuple of 2 int
        Spatial size of output image for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 upscale_out_size):
        super(ASPPAvgBranch, self).__init__()
        self.upscale_out_size = upscale_out_size

        with self.init_scope():
            self.pool = partial(
                F.average_pooling_2d,
                ksize=1)
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        in_size = self.upscale_out_size if self.upscale_out_size is not None else x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.resize_images(x, output_shape=in_size)
        return x


class AtrousSpatialPyramidPooling(Chain):
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
                 upscale_out_size):
        super(AtrousSpatialPyramidPooling, self).__init__()
        atrous_rates = [12, 24, 36]
        assert (in_channels % 8 == 0)
        mid_channels = in_channels // 8
        project_in_channels = 5 * mid_channels

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels))
                for i, atrous_rate in enumerate(atrous_rates):
                    setattr(self.branches, "branch{}".format(i + 2), conv3x3_block(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        pad=atrous_rate,
                        dilate=atrous_rate))
                setattr(self.branches, "branch5", ASPPAvgBranch(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    upscale_out_size=upscale_out_size))
            self.conv = conv1x1_block(
                in_channels=project_in_channels,
                out_channels=mid_channels)
            self.dropout = partial(
                F.dropout,
                ratio=0.5)

    def __call__(self, x):
        x = self.branches(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class DeepLabv3(Chain):
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
                 classes=21):
        super(DeepLabv3, self).__init__()
        assert (in_channels > 0)
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size

        with self.init_scope():
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

    def __call__(self, x):
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
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create DeepLabv3 model with specific parameters.

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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_voc",
                         **kwargs)


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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_voc",
                         **kwargs)


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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd152b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd152b_voc",
                         **kwargs)


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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_coco",
                         **kwargs)


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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_coco",
                         **kwargs)


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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd152b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd152b_coco",
                         **kwargs)


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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_cityscapes",
                         **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

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
        weight_count = net.count_params()
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

        x = np.zeros((1, 3, in_size[0], in_size[1]), np.float32)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
