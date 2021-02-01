"""
    CenterNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Objects as Points,' https://arxiv.org/abs/1904.07850.
"""

__all__ = ['CenterNet', 'centernet_resnet18_voc', 'centernet_resnet18_coco', 'centernet_resnet50b_voc',
           'centernet_resnet50b_coco', 'centernet_resnet101b_voc', 'centernet_resnet101b_coco',
           'CenterNetHeatmapMaxDet']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, conv1x1, conv3x3_block, DeconvBlock, Concurrent, SimpleSequential, is_channels_first
from .resnet import resnet18, resnet50b, resnet101b


class CenterNetDecoderUnit(nn.Layer):
    """
    CenterNet decoder unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(CenterNetDecoderUnit, self).__init__(**kwargs)
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.deconv = DeconvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            strides=2,
            padding=1,
            data_format=data_format,
            name="deconv")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.deconv(x, training=training)
        return x


class CenterNetHeadBlock(nn.Layer):
    """
    CenterNet simple head block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(CenterNetHeadBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            use_bias=True,
            use_bn=False,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterNetHeatmapBlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 do_nms,
                 data_format="channels_last",
                 **kwargs):
        super(CenterNetHeatmapBlock, self).__init__(**kwargs)
        self.do_nms = do_nms

        self.head = CenterNetHeadBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="head")
        self.sigmoid = tf.nn.sigmoid
        if self.do_nms:
            self.pool = MaxPool2d(
                pool_size=3,
                strides=1,
                padding=1,
                data_format=data_format,
                name="pool")

    def call(self, x, training=None):
        x = self.head(x)
        x = self.sigmoid(x)
        if self.do_nms:
            y = self.pool(x)
            x = x * (y.numpy() == x.numpy())
        else:
            eps = 1e-4
            x = tf.clip_by_value(x, clip_value_min=eps, clip_value_max=(1.0 - eps))
        return x


class CenterNetHeatmapMaxDet(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 topk=40,
                 scale=4,
                 max_batch=256,
                 data_format="channels_last",
                 **kwargs):
        super(CenterNetHeatmapMaxDet, self).__init__(**kwargs)
        self.topk = topk
        self.scale = scale
        self.max_batch = max_batch
        self.data_format = data_format

    def call(self, x, training=None):
        import numpy as np

        x_ = x.numpy()
        if not is_channels_first(self.data_format):
            x_ = x_.transpose((0, 3, 1, 2))

        heatmap = x_[:, :-4]
        wh = x_[:, -4:-2]
        reg = x_[:, -2:]
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
        bboxes = tf.stack((topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h), axis=-1)

        bboxes = bboxes * self.scale
        topk_classes = tf.expand_dims(topk_classes, axis=-1)
        scores = tf.expand_dims(scores, axis=-1)
        result = tf.concat((bboxes, topk_classes, scores), axis=-1)
        return result


class CenterNet(tf.keras.Model):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
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
                 data_format="channels_last",
                 **kwargs):
        super(CenterNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.in_channels = in_channels
        self.return_heatmap = return_heatmap
        self.data_format = data_format

        self.backbone = backbone
        self.backbone._name = "backbone"

        self.decoder = SimpleSequential(name="decoder")
        in_channels = backbone_out_channels
        for i, out_channels in enumerate(channels):
            self.decoder.add(CenterNetDecoderUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                data_format=data_format,
                name="unit{}".format(i + 1)))
            in_channels = out_channels

        heads = Concurrent(
            data_format=data_format,
            name="heads")
        heads.add(CenterNetHeatmapBlock(
            in_channels=in_channels,
            out_channels=classes,
            do_nms=(not self.return_heatmap),
            data_format=data_format,
            name="heapmap_block"))
        heads.add(CenterNetHeadBlock(
            in_channels=in_channels,
            out_channels=2,
            data_format=data_format,
            name="wh_block"))
        heads.add(CenterNetHeadBlock(
            in_channels=in_channels,
            out_channels=2,
            data_format=data_format,
            name="reg_block"))
        self.decoder.add(heads)

        if not self.return_heatmap:
            self.heatmap_max_det = CenterNetHeatmapMaxDet(
                topk=topk,
                scale=4,
                data_format=data_format,
                name="heatmap_max_det")

    def call(self, x, training=None):
        x = self.backbone(x, training=training)
        x = self.decoder(x, training=training)
        if not self.return_heatmap or not tf.executing_eagerly():
            x = self.heatmap_max_det(x)
        return x


def get_centernet(backbone,
                  backbone_out_channels,
                  classes,
                  model_name=None,
                  data_format="channels_last",
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
        data_format=data_format,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def centernet_resnet18_voc(pretrained_backbone=False, classes=20, data_format="channels_last", **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 20
        Number of classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone.children[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=512, classes=classes,
                         model_name="centernet_resnet18_voc", data_format=data_format, **kwargs)


def centernet_resnet18_coco(pretrained_backbone=False, classes=80, data_format="channels_last", **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 80
        Number of classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone.children[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=512, classes=classes,
                         model_name="centernet_resnet18_coco", data_format=data_format, **kwargs)


def centernet_resnet50b_voc(pretrained_backbone=False, classes=20, data_format="channels_last", **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 20
        Number of classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone.children[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet50b_voc", data_format=data_format, **kwargs)


def centernet_resnet50b_coco(pretrained_backbone=False, classes=80, data_format="channels_last", **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 80
        Number of classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone.children[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet50b_coco", data_format=data_format, **kwargs)


def centernet_resnet101b_voc(pretrained_backbone=False, classes=20, data_format="channels_last", **kwargs):
    """
    CenterNet model on the base of ResNet-101b for VOC Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 20
        Number of classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone.children[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet101b_voc", data_format=data_format, **kwargs)


def centernet_resnet101b_coco(pretrained_backbone=False, classes=80, data_format="channels_last", **kwargs):
    """
    CenterNet model on the base of ResNet-101b for COCO Detection from 'Objects as Points,'
    https://arxiv.org/abs/1904.07850.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 80
        Number of classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone.children[-1]
    return get_centernet(backbone=backbone, backbone_out_channels=2048, classes=classes,
                         model_name="centernet_resnet101b_coco", data_format=data_format, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
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

        net = model(pretrained=pretrained, topk=topk, in_size=in_size, return_heatmap=return_heatmap,
                    data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (y.shape[0] == batch)
        if return_heatmap:
            if is_channels_first(data_format):
                assert (y.shape[1] == classes + 4) and (y.shape[2] == x.shape[2] // 4) and (
                            y.shape[3] == x.shape[3] // 4)
            else:
                assert (y.shape[3] == classes + 4) and (y.shape[1] == x.shape[1] // 4) and (
                            y.shape[2] == x.shape[2] // 4)
        else:
            assert (y.shape[1] == topk) and (y.shape[2] == 6)

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != centernet_resnet18_voc or weight_count == 14215640)
        assert (model != centernet_resnet18_coco or weight_count == 14219540)
        assert (model != centernet_resnet50b_voc or weight_count == 30086104)
        assert (model != centernet_resnet50b_coco or weight_count == 30090004)
        assert (model != centernet_resnet101b_voc or weight_count == 49078232)
        assert (model != centernet_resnet101b_coco or weight_count == 49082132)


if __name__ == "__main__":
    _test()
