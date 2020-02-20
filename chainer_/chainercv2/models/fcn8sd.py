"""
    FCN-8s(d) for image segmentation, implemented in Chainer.
    Original paper: 'Fully Convolutional Networks for Semantic Segmentation,' https://arxiv.org/abs/1411.4038.
"""

__all__ = ['FCN8sd', 'fcn8sd_resnetd50b_voc', 'fcn8sd_resnetd101b_voc', 'fcn8sd_resnetd50b_coco',
           'fcn8sd_resnetd101b_coco', 'fcn8sd_resnetd50b_ade20k', 'fcn8sd_resnetd101b_ade20k',
           'fcn8sd_resnetd50b_cityscapes', 'fcn8sd_resnetd101b_cityscapes']

import os
import chainer.functions as F
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv3x3_block
from .resnetd import resnetd50b, resnetd101b


class FCNFinalBlock(Chain):
    """
    FCN-8s(d) final block.

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
        super(FCNFinalBlock, self).__init__()
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


class FCN8sd(Chain):
    """
    FCN-8s(d) model from 'Fully Convolutional Networks for Semantic Segmentation,' https://arxiv.org/abs/1411.4038.
    It is an experimental model mixed FCN-8s and PSPNet.

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
        super(FCN8sd, self).__init__()
        assert (in_channels > 0)
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size

        with self.init_scope():
            self.backbone = backbone
            pool_out_channels = backbone_out_channels
            self.final_block = FCNFinalBlock(
                in_channels=pool_out_channels,
                out_channels=classes)
            if self.aux:
                aux_out_channels = backbone_out_channels // 2
                self.aux_block = FCNFinalBlock(
                    in_channels=aux_out_channels,
                    out_channels=classes)

    def forward(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x, y = self.backbone(x)
        x = self.final_block(x, in_size)
        if self.aux:
            y = self.aux_block(y, in_size)
            return x, y
        else:
            return x


def get_fcn8sd(backbone,
               classes,
               aux=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create FCN-8s(d) model with specific parameters.

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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    net = FCN8sd(
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


def fcn8sd_resnetd50b_voc(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-50b for Pascal VOC from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd50b_voc", **kwargs)


def fcn8sd_resnetd101b_voc(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-101b for Pascal VOC from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd101b_voc",
                      **kwargs)


def fcn8sd_resnetd50b_coco(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-50b for COCO from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd50b_coco",
                      **kwargs)


def fcn8sd_resnetd101b_coco(pretrained_backbone=False, classes=21, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-101b for COCO from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd101b_coco",
                      **kwargs)


def fcn8sd_resnetd50b_ade20k(pretrained_backbone=False, classes=150, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-50b for ADE20K from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd50b_ade20k",
                      **kwargs)


def fcn8sd_resnetd101b_ade20k(pretrained_backbone=False, classes=150, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-101b for ADE20K from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd101b_ade20k",
                      **kwargs)


def fcn8sd_resnetd50b_cityscapes(pretrained_backbone=False, classes=19, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-50b for Cityscapes from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd50b_cityscapes",
                      **kwargs)


def fcn8sd_resnetd101b_cityscapes(pretrained_backbone=False, classes=19, aux=True, **kwargs):
    """
    FCN-8s(d) model on the base of ResNet(D)-101b for Cityscapes from 'Fully Convolutional Networks for Semantic
    Segmentation,' https://arxiv.org/abs/1411.4038.

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
    return get_fcn8sd(backbone=backbone, classes=classes, aux=aux, model_name="fcn8sd_resnetd101b_cityscapes",
                      **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (480, 480)
    aux = False
    pretrained = False

    models = [
        (fcn8sd_resnetd50b_voc, 21),
        (fcn8sd_resnetd101b_voc, 21),
        (fcn8sd_resnetd50b_coco, 21),
        (fcn8sd_resnetd101b_coco, 21),
        (fcn8sd_resnetd50b_ade20k, 150),
        (fcn8sd_resnetd101b_ade20k, 150),
        (fcn8sd_resnetd50b_cityscapes, 19),
        (fcn8sd_resnetd101b_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != fcn8sd_resnetd50b_voc or weight_count == 35445994)
            assert (model != fcn8sd_resnetd101b_voc or weight_count == 54438122)
            assert (model != fcn8sd_resnetd50b_coco or weight_count == 35445994)
            assert (model != fcn8sd_resnetd101b_coco or weight_count == 54438122)
            assert (model != fcn8sd_resnetd50b_ade20k or weight_count == 35545324)
            assert (model != fcn8sd_resnetd101b_ade20k or weight_count == 54537452)
            assert (model != fcn8sd_resnetd50b_cityscapes or weight_count == 35444454)
            assert (model != fcn8sd_resnetd101b_cityscapes or weight_count == 54436582)
        else:
            assert (model != fcn8sd_resnetd50b_voc or weight_count == 33080789)
            assert (model != fcn8sd_resnetd101b_voc or weight_count == 52072917)
            assert (model != fcn8sd_resnetd50b_coco or weight_count == 33080789)
            assert (model != fcn8sd_resnetd101b_coco or weight_count == 52072917)
            assert (model != fcn8sd_resnetd50b_ade20k or weight_count == 33146966)
            assert (model != fcn8sd_resnetd101b_ade20k or weight_count == 52139094)
            assert (model != fcn8sd_resnetd50b_cityscapes or weight_count == 33079763)
            assert (model != fcn8sd_resnetd101b_cityscapes or weight_count == 52071891)

        x = np.zeros((1, 3, in_size[0], in_size[1]), np.float32)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
