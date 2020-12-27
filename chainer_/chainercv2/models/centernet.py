"""
    CenterNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Objects as Points,' https://arxiv.org/abs/1904.07850.
"""

__all__ = ['CenterNet', 'centernet_resnet18_voc', 'centernet_resnet18_coco', 'centernet_resnet50b_voc',
           'centernet_resnet50b_coco', 'centernet_resnet101b_voc', 'centernet_resnet101b_coco',
           'CenterNetHeatmapMaxDet']

import os
import chainer.functions as F
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv3x3_block, DeconvBlock, Concurrent, SimpleSequential
from .resnet import resnet18, resnet50b, resnet101b


class CenterNetDecoderUnit(Chain):
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
                 out_channels,
                 **kwargs):
        super(CenterNetDecoderUnit, self).__init__(**kwargs)
        with self.init_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True)
            self.deconv = DeconvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                ksize=4,
                stride=2,
                pad=1)

    def __call__(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


class CenterNetHeadBlock(Chain):
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
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                use_bias=True,
                use_bn=False)
            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterNetHeatmapBlock(Chain):
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

        with self.init_scope():
            self.head = CenterNetHeadBlock(
                in_channels=in_channels,
                out_channels=out_channels)
            self.sigmoid = F.sigmoid
            if self.do_nms:
                self.pool = partial(
                    F.max_pooling_2d,
                    ksize=3,
                    stride=1,
                    pad=1,
                    cover_all=False)

    def __call__(self, x):
        x = self.head(x)
        x = self.sigmoid(x)
        if self.do_nms:
            y = self.pool(x)
            x = x * (y.array == x.array)
        else:
            eps = 1e-4
            x = F.clip(x, x_min=eps, x_max=(1.0 - eps))
        return x


class CenterNetHeatmapMaxDet(Chain):
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

    def __call__(self, x):
        import numpy as np

        heatmap = x[:, :-4].array
        wh = x[:, -4:-2].array
        reg = x[:, -2:].array
        batch, _, out_h, out_w = heatmap.shape

        heatmap_flat = heatmap.reshape((batch, -1))
        indices = np.argsort(heatmap_flat)[:, -self.topk:]
        scores = np.take_along_axis(heatmap_flat, indices=indices, axis=-1)
        topk_classes = (indices // (out_h * out_w)).astype(dtype=np.float32)
        topk_indices = indices % (out_h * out_w)
        topk_ys = (topk_indices // out_w).astype(dtype=np.float32)
        topk_xs = (topk_indices % out_w).astype(dtype=np.float32)
        center = reg.transpose((0, 2, 3, 1)).reshape((batch, -1, 2))
        wh = wh.transpose((0, 2, 3, 1)).reshape((batch, -1, 2))
        xs = np.take_along_axis(center[:, :, 0], indices=topk_indices, axis=-1)
        ys = np.take_along_axis(center[:, :, 1], indices=topk_indices, axis=-1)
        topk_xs = topk_xs + xs
        topk_ys = topk_ys + ys
        w = np.take_along_axis(wh[:, :, 0], indices=topk_indices, axis=-1)
        h = np.take_along_axis(wh[:, :, 1], indices=topk_indices, axis=-1)
        half_w = 0.5 * w
        half_h = 0.5 * h
        bboxes = F.stack((topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h), axis=-1)

        bboxes = bboxes * self.scale
        topk_classes = F.expand_dims(topk_classes, axis=-1)
        scores = F.expand_dims(scores, axis=-1)
        result = F.concat((bboxes, topk_classes, scores), axis=-1)
        return result


class CenterNet(Chain):
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
    classes : int, default 80
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
                 classes=80,
                 **kwargs):
        super(CenterNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.in_channels = in_channels
        self.return_heatmap = return_heatmap

        with self.init_scope():
            self.backbone = backbone

            self.decoder = SimpleSequential()
            with self.decoder.init_scope():
                in_channels = backbone_out_channels
                for i, out_channels in enumerate(channels):
                    setattr(self.decoder, "unit{}".format(i + 1), CenterNetDecoderUnit(
                        in_channels=in_channels,
                        out_channels=out_channels))
                    in_channels = out_channels

                heads = Concurrent()
                with heads.init_scope():
                    setattr(heads, "heapmap_block", CenterNetHeatmapBlock(
                        in_channels=in_channels,
                        out_channels=classes,
                        do_nms=(not self.return_heatmap)))
                    setattr(heads, "wh_block", CenterNetHeadBlock(
                        in_channels=in_channels,
                        out_channels=2))
                    setattr(heads, "reg_block", CenterNetHeadBlock(
                        in_channels=in_channels,
                        out_channels=2))
                    setattr(self.decoder, "heads", heads)

            if not self.return_heatmap:
                self.heatmap_max_det = CenterNetHeatmapMaxDet(
                    topk=topk,
                    scale=4)

    def __call__(self, x):
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
                  root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone.final_pool
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone.final_pool
    return get_centernet(backbone=backbone, backbone_out_channels=512, classes=classes,
                         model_name='centernet_resnet18_coco', **kwargs)


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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone.final_pool
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone.final_pool
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone.final_pool
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone.final_pool
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet101b_coco", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

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
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != centernet_resnet18_voc or weight_count == 14215640)
        assert (model != centernet_resnet18_coco or weight_count == 14219540)
        assert (model != centernet_resnet50b_voc or weight_count == 30086104)
        assert (model != centernet_resnet50b_coco or weight_count == 30090004)
        assert (model != centernet_resnet101b_voc or weight_count == 49078232)
        assert (model != centernet_resnet101b_coco or weight_count == 49082132)

        batch = 14
        x = np.random.rand(batch, 3, in_size[0], in_size[1]).astype(np.float32)
        y = net(x)
        assert (y.shape[0] == batch)
        if return_heatmap:
            assert (y.shape[1] == classes + 4) and (y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4)
        else:
            assert (y.shape[1] == topk) and (y.shape[2] == 6)


if __name__ == "__main__":
    _test()
