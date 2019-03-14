"""
    PSPNet, implemented in Gluon.
    Original paper: 'Pyramid Scene Parsing Network,' https://arxiv.org/abs/1612.01105.
"""

__all__ = ['PSPNet', 'pspnet_resnet50_voc', 'pspnet_resnet101_voc', 'pspnet_resnet50_ade20k']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity
from common import conv1x1, conv1x1_block, conv3x3_block
from resnet import resnet50, resnet101


class PyramidPoolingBranch(HybridBlock):
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
    in_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 output_size,
                 in_size,
                 **kwargs):
        super(PyramidPoolingBranch, self).__init__(**kwargs)
        self.output_size = output_size
        self.in_size = in_size

        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = F.contrib.AdaptiveAvgPooling2D(x, output_size=self.output_size)
        x = self.conv(x)
        x = F.contrib.BilinearResize2D(x, height=self.in_size[0], width=self.in_size[1])
        return x


class PyramidPooling(HybridBlock):
    """
    Pyramid Pooling module.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    in_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 in_size,
                 **kwargs):
        super(PyramidPooling, self).__init__(**kwargs)
        output_sizes = [1, 2, 3, 6]
        assert (len(output_sizes) == 4)
        assert (in_channels % 4 == 0)
        mid_channels = in_channels // 4

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix='')
            self.branches.add(Identity())
            for i, output_size in enumerate(output_sizes):
                self.branches.add(PyramidPoolingBranch(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    output_size=output_size,
                    in_size=in_size))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class PSPNet(HybridBlock):
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
    classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=21,
                 **kwargs):
        super(PSPNet, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        pool_out_channels = 2 * backbone_out_channels
        mid_channels = backbone_out_channels // 4

        with self.name_scope():
            self.backbone = backbone
            self.pool = PyramidPooling(
                in_channels=backbone_out_channels,
                in_size=(self.in_size[0] // 8, self.in_size[1] // 8))
            self.conv1 = conv3x3_block(
                in_channels=pool_out_channels,
                out_channels=mid_channels)
            self.dropout = nn.Dropout(rate=0.1)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=classes)

    def hybrid_forward(self, F, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.contrib.BilinearResize2D(x, height=self.in_size[0], width=self.in_size[1])
        return x


def get_pspnet(backbone,
               classes,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join('~', '.mxnet', 'models'),
               **kwargs):
    """
    Create PSPNet model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    net = PSPNet(
        backbone=backbone,
        classes=classes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def pspnet_resnet50_voc(pretrained_backbone=False, classes=21, **kwargs):
    """
    PSPNet model on the base of ResNet-50 for Pascal VOC from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50(pretrained=pretrained_backbone).features[:-1]
    return get_pspnet(backbone=backbone, classes=classes, model_name="pspnet_resnet50_voc", **kwargs)


def pspnet_resnet101_voc(pretrained_backbone=False, classes=21, **kwargs):
    """
    PSPNet model on the base of ResNet-101 for Pascal VOC from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101(pretrained=pretrained_backbone).features[:-1]
    return get_pspnet(backbone=backbone, classes=classes, model_name="pspnet_resnet101_voc", **kwargs)


def pspnet_resnet50_ade20k(pretrained_backbone=False, classes=150, **kwargs):
    """
    PSPNet model on the base of ResNet-50 for ADE20K from 'Pyramid Scene Parsing Network,'
    https://arxiv.org/abs/1612.01105.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 150
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50(pretrained=pretrained_backbone).features[:-1]
    return get_pspnet(backbone=backbone, classes=classes, model_name="pspnet_resnet50_ade20k", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (pspnet_resnet50_voc, 21),
        (pspnet_resnet101_voc, 21),
        (pspnet_resnet50_ade20k, 150),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

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
        assert (model != pspnet_resnet50_voc or weight_count == 46592576)
        assert (model != pspnet_resnet101_voc or weight_count == 65584704)
        assert (model != pspnet_resnet50_ade20k or weight_count == 46658624)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert ((y.size(0) == x.size(0)) and (y.size(1) == num_classes) and (y.size(2) == x.size(2)) and
                (y.size(3) == x.size(3)))


if __name__ == "__main__":
    _test()
