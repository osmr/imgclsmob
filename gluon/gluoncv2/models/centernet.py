"""
    CenterNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Objects as Points,' https://arxiv.org/abs/1904.07850.
"""

__all__ = ['CenterNet', 'centernet_resnet18_voc', 'centernet_resnet18_coco', 'centernet_resnet50b_voc',
           'centernet_resnet50b_coco', 'centernet_resnet101b_voc', 'centernet_resnet101b_coco',
           'CenterNetHeatmapMaxDet']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import conv1x1, conv3x3_block, DeconvBlock
from .resnet import resnet18, resnet50b, resnet101b


class CenterNetDecoderUnit(HybridBlock):
    """
    CenterNet decoder unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(CenterNetDecoderUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_use_global_stats=bn_use_global_stats)
            self.deconv = DeconvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                strides=2,
                padding=1,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


class CenterNetHeadBlock(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(CenterNetHeadBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                use_bias=True,
                use_bn=False)
            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterNetHeatmapBlock(HybridBlock):
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
                 do_nms,
                 **kwargs):
        super(CenterNetHeatmapBlock, self).__init__(**kwargs)
        self.do_nms = do_nms

        with self.name_scope():
            self.head = CenterNetHeadBlock(
                in_channels=in_channels,
                out_channels=out_channels)
            self.sigmoid = nn.Activation("sigmoid")
            if self.do_nms:
                self.pool = nn.MaxPool2D(
                    pool_size=3,
                    strides=1,
                    padding=1)

    def hybrid_forward(self, F, x):
        x = self.head(x)
        x = self.sigmoid(x)
        if self.do_nms:
            y = self.pool(x)
            x = x * F.broadcast_equal(y, x)
        else:
            eps = 1e-4
            x = x.clip(a_min=eps, a_max=(1.0 - eps))
        return x


class CenterNetHeatmapMaxDet(HybridBlock):
    """
    CenterNet decoder for heads (heatmap, wh, reg).

    Parameters:
    ----------
    topk : int, default 40
        Keep only `topk` detections.
    scale : int, default is 4
        Downsampling scale factor.
    max_batch : int, default is 256
        Maximal batch size.
    """
    def __init__(self,
                 topk=40,
                 scale=4,
                 max_batch=256,
                 **kwargs):
        super(CenterNetHeatmapMaxDet, self).__init__(**kwargs)
        self.topk = topk
        self.scale = scale
        self.max_batch = max_batch

    def hybrid_forward(self, F, x):
        heatmap = x.slice_axis(axis=1, begin=0, end=-4)
        wh = x.slice_axis(axis=1, begin=-4, end=-2)
        reg = x.slice_axis(axis=1, begin=-2, end=None)
        _, _, out_h, out_w = heatmap.shape_array().split(num_outputs=4, axis=0)
        scores, indices = heatmap.reshape((0, -1)).topk(k=self.topk, ret_typ="both")
        indices = indices.astype(dtype="int64")
        topk_classes = F.broadcast_div(indices, (out_h * out_w)).astype(dtype="float32")
        topk_indices = F.broadcast_mod(indices, (out_h * out_w))
        topk_ys = F.broadcast_div(topk_indices, out_w).astype(dtype="float32")
        topk_xs = F.broadcast_mod(topk_indices, out_w).astype(dtype="float32")
        center = reg.transpose((0, 2, 3, 1)).reshape((0, -1, 2))
        wh = wh.transpose((0, 2, 3, 1)).reshape((0, -1, 2))
        batch_indices = F.arange(self.max_batch).slice_like(center, axes=0).expand_dims(-1).repeat(self.topk, 1).\
            astype(dtype="int64")
        reg_xs_indices = F.zeros_like(batch_indices, dtype="int64")
        reg_ys_indices = F.ones_like(batch_indices, dtype="int64")
        reg_xs = F.concat(batch_indices, topk_indices, reg_xs_indices, dim=0).reshape((3, -1))
        reg_ys = F.concat(batch_indices, topk_indices, reg_ys_indices, dim=0).reshape((3, -1))
        xs = F.gather_nd(center, reg_xs).reshape((-1, self.topk))
        ys = F.gather_nd(center, reg_ys).reshape((-1, self.topk))
        topk_xs = topk_xs + xs
        topk_ys = topk_ys + ys
        w = F.gather_nd(wh, reg_xs).reshape((-1, self.topk))
        h = F.gather_nd(wh, reg_ys).reshape((-1, self.topk))
        half_w = 0.5 * w
        half_h = 0.5 * h
        bboxes = F.stack(topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h, axis=-1)

        bboxes = bboxes * self.scale
        topk_classes = topk_classes.expand_dims(axis=-1)
        scores = scores.expand_dims(axis=-1)
        result = F.concat(bboxes, topk_classes, scores, dim=-1)
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


class CenterNet(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    return_heatmap : bool, default False
        Whether to return only heatmap.
    topk : int, default 40
        Keep only `topk` detections.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (512, 512)
        Spatial size of the expected input image.
    classes : int, default 80
        Number of classification classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels,
                 channels,
                 bn_use_global_stats=False,
                 return_heatmap=False,
                 topk=40,
                 in_channels=3,
                 in_size=(512, 512),
                 classes=80,
                 **kwargs):
        super(CenterNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.in_channels = in_channels
        self.return_heatmap = return_heatmap

        with self.name_scope():
            self.backbone = backbone

            self.decoder = nn.HybridSequential(prefix="")
            in_channels = backbone_out_channels
            for out_channels in channels:
                self.decoder.add(CenterNetDecoderUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = out_channels

            heads = HybridConcurrent(axis=1, prefix="")
            heads.add(CenterNetHeatmapBlock(
                in_channels=in_channels,
                out_channels=classes,
                do_nms=(not self.return_heatmap)))
            heads.add(CenterNetHeadBlock(
                in_channels=in_channels,
                out_channels=2))
            heads.add(CenterNetHeadBlock(
                in_channels=in_channels,
                out_channels=2))
            self.decoder.add(heads)

            if not self.return_heatmap:
                self.heatmap_max_det = CenterNetHeatmapMaxDet(
                    topk=topk,
                    scale=4)

    def hybrid_forward(self, F, x):
        x = self.backbone(x)
        x = self.decoder(x)
        if not self.return_heatmap:
            x = self.heatmap_max_det(x)
        return x


def get_centernet(backbone,
                  backbone_out_channels,
                  classes,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create CenterNet model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    classes : int
        Number of classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns:
    -------
    HybridBlock
        A network.
    """
    channels = [256, 128, 64]

    net = CenterNet(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
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


def centernet_resnet18_voc(pretrained_backbone=False, classes=20, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 20
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features[:-1]
    return get_centernet(backbone=backbone, backbone_out_channels=512, classes=classes,
                         model_name="centernet_resnet18_voc", **kwargs)


def centernet_resnet18_coco(pretrained_backbone=False, classes=80, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 80
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features[:-1]
    return get_centernet(backbone=backbone, backbone_out_channels=512, classes=classes,
                         model_name="centernet_resnet18_coco", **kwargs)


def centernet_resnet50b_voc(pretrained_backbone=False, classes=20, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 20
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features[:-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet50b_voc", **kwargs)


def centernet_resnet50b_coco(pretrained_backbone=False, classes=80, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 80
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features[:-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet50b_coco", **kwargs)


def centernet_resnet101b_voc(pretrained_backbone=False, classes=20, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 20
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features[:-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet101b_voc", **kwargs)


def centernet_resnet101b_coco(pretrained_backbone=False, classes=80, **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 80
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features[:-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet101b_coco", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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
        assert (model != centernet_resnet18_voc or weight_count == 14215640)
        assert (model != centernet_resnet18_coco or weight_count == 14219540)
        assert (model != centernet_resnet50b_voc or weight_count == 30086104)
        assert (model != centernet_resnet50b_coco or weight_count == 30090004)
        assert (model != centernet_resnet101b_voc or weight_count == 49078232)
        assert (model != centernet_resnet101b_coco or weight_count == 49082132)

        batch = 14
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (y.shape[0] == batch)
        if return_heatmap:
            assert (y.shape[1] == classes + 4) and (y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4)
        else:
            assert (y.shape[1] == topk) and (y.shape[2] == 6)


if __name__ == "__main__":
    _test()
