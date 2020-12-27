"""
    NTS-Net for CUB-200-2011, implemented in Chainer.
    Original paper: 'Learning to Navigate for Fine-grained Classification,' https://arxiv.org/abs/1809.00287.
"""

__all__ = ['NTSNet', 'ntsnet_cub']

import os
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer.serializers import load_npz
from functools import partial
from .common import conv1x1, conv3x3, Flatten, AdaptiveAvgPool2D, SimpleSequential
from .resnet import resnet50b


def hard_nms(cdds,
             top_n=10,
             iou_thresh=0.25):
    """
    Hard Non-Maximum Suppression.

    Parameters:
    ----------
    cdds : np.array
        Borders.
    top_n : int, default 10
        Number of top-K informative regions.
    iou_thresh : float, default 0.25
        IoU threshold.

    Returns:
    -------
    np.array
        Filtered borders.
    """
    assert (type(cdds) == np.ndarray)
    assert (len(cdds.shape) == 2)
    assert (cdds.shape[1] >= 5)

    cdds = cdds.copy()
    indices = np.argsort(cdds[:, 0])
    cdds = cdds[indices]
    cdd_results = []

    res = cdds

    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == top_n:
            return np.array(cdd_results)
        res = res[:-1]

        start_max = np.maximum(res[:, 1:3], cdd[1:3])
        end_min = np.minimum(res[:, 3:5], cdd[3:5])
        lengths = end_min - start_max
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (
            cdd[4] - cdd[2]) - intersec_map)
        res = res[iou_map_cur < iou_thresh]

    return np.array(cdd_results)


class NavigatorBranch(Chain):
    """
    Navigator branch block for Navigator unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(NavigatorBranch, self).__init__()
        mid_channels = 128

        with self.init_scope():
            self.down_conv = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride,
                use_bias=True)
            self.activ = F.relu
            self.tidy_conv = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)
            self.flatten = Flatten()

    def __call__(self, x):
        y = self.down_conv(x)
        y = self.activ(y)
        z = self.tidy_conv(y)
        z = self.flatten(z)
        return z, y


class NavigatorUnit(Chain):
    """
    Navigator init.
    """
    def __init__(self):
        super(NavigatorUnit, self).__init__()
        with self.init_scope():
            self.branch1 = NavigatorBranch(
                in_channels=2048,
                out_channels=6,
                stride=1)
            self.branch2 = NavigatorBranch(
                in_channels=128,
                out_channels=6,
                stride=2)
            self.branch3 = NavigatorBranch(
                in_channels=128,
                out_channels=9,
                stride=2)

    def __call__(self, x):
        t1, x = self.branch1(x)
        t2, x = self.branch2(x)
        t3, _ = self.branch3(x)
        return F.concat((t1, t2, t3), axis=1)


class NTSNet(Chain):
    """
    NTS-Net model from 'Learning to Navigate for Fine-grained Classification,' https://arxiv.org/abs/1809.00287.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    aux : bool, default False
        Whether to output auxiliary results.
    top_n : int, default 4
        Number of extra top-K informative regions.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 backbone,
                 aux=False,
                 top_n=4,
                 in_channels=3,
                 in_size=(448, 448),
                 classes=200):
        super(NTSNet, self).__init__()
        assert (in_channels > 0)
        self.in_size = in_size
        self.classes = classes
        pad_side = 224
        self.pad_width = ((0, 0), (0, 0), (pad_side, pad_side), (pad_side, pad_side))

        self.top_n = top_n
        self.aux = aux
        self.num_cat = 4

        _, edge_anchors, _ = self._generate_default_anchor_maps()
        self.edge_anchors = (edge_anchors + 224).astype(np.int)
        self.edge_anchors = np.concatenate(
            (self.edge_anchors.copy(), np.arange(0, len(self.edge_anchors)).reshape(-1, 1)), axis=1)

        with self.init_scope():
            self.backbone = backbone

            self.backbone_tail = SimpleSequential()
            with self.backbone_tail.init_scope():
                setattr(self.backbone_tail, "final_pool", AdaptiveAvgPool2D())
                setattr(self.backbone_tail, "flatten", Flatten())
                setattr(self.backbone_tail, "dropout", partial(
                    F.dropout,
                    ratio=0.5))
            self.backbone_classifier = L.Linear(
                in_size=(512 * 4),
                out_size=classes)

            self.navigator_unit = NavigatorUnit()
            self.concat_net = L.Linear(
                in_size=(2048 * (self.num_cat + 1)),
                out_size=classes)

            if self.aux:
                self.partcls_net = L.Linear(
                    in_size=(512 * 4),
                    out_size=classes)

    def __call__(self, x):
        raw_pre_features = self.backbone(x)

        rpn_score = self.navigator_unit(raw_pre_features)
        rpn_score.to_cpu()
        all_cdds = [np.concatenate((y.reshape(-1, 1), self.edge_anchors.copy()), axis=1)
                    for y in rpn_score.array]
        top_n_cdds = [hard_nms(y, top_n=self.top_n, iou_thresh=0.25) for y in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int64)
        top_n_index = np.array(top_n_index, dtype=np.int64)
        top_n_prob = np.take_along_axis(rpn_score.array, top_n_index, axis=1)

        batch = x.shape[0]
        x_pad = F.pad(x, pad_width=self.pad_width, mode="constant", constant_values=0)
        part_imgs = []
        for i in range(batch):
            for j in range(self.top_n):
                y0, x0, y1, x1 = tuple(top_n_cdds[i][j, 1:5].astype(np.int64))
                x_res = F.resize_images(
                    x_pad[i:i + 1, :, y0:y1, x0:x1],
                    output_shape=(224, 224))
                part_imgs.append(x_res)
        part_imgs = F.concat(tuple(part_imgs), axis=0)
        part_features = self.backbone_tail(self.backbone(part_imgs))

        part_feature = part_features.reshape((batch, self.top_n, -1))
        part_feature = part_feature[:, :self.num_cat, :]
        part_feature = part_feature.reshape((batch, -1))

        raw_features = self.backbone_tail(raw_pre_features)

        concat_out = F.concat((part_feature, raw_features), axis=1)
        concat_logits = self.concat_net(concat_out)

        if self.aux:
            raw_logits = self.backbone_classifier(raw_features)
            part_logits = self.partcls_net(part_features).reshape((batch, self.top_n, -1))
            return concat_logits, raw_logits, part_logits, top_n_prob
        else:
            return concat_logits

    @staticmethod
    def _generate_default_anchor_maps(input_shape=(448, 448)):
        """
        Generate default anchor maps.

        Parameters:
        ----------
        input_shape : tuple of 2 int
            Input image size.

        Returns:
        -------
        center_anchors : np.array
            anchors * 4 (oy, ox, h, w).
        edge_anchors : np.array
            anchors * 4 (y0, x0, y1, x1).
        anchor_area : np.array
            anchors * 1 (area).
        """
        anchor_scale = [2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        anchor_aspect_ratio = [0.667, 1, 1.5]
        anchors_setting = (
            dict(layer="p3", stride=32, size=48, scale=anchor_scale, aspect_ratio=anchor_aspect_ratio),
            dict(layer="p4", stride=64, size=96, scale=anchor_scale, aspect_ratio=anchor_aspect_ratio),
            dict(layer="p5", stride=128, size=192, scale=[1, anchor_scale[0], anchor_scale[1]],
                 aspect_ratio=anchor_aspect_ratio),
        )

        center_anchors = np.zeros((0, 4), dtype=np.float32)
        edge_anchors = np.zeros((0, 4), dtype=np.float32)
        anchor_areas = np.zeros((0,), dtype=np.float32)
        input_shape = np.array(input_shape, dtype=int)

        for anchor_info in anchors_setting:

            stride = anchor_info["stride"]
            size = anchor_info["size"]
            scales = anchor_info["scale"]
            aspect_ratios = anchor_info["aspect_ratio"]

            output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
            output_map_shape = output_map_shape.astype(np.int)
            output_shape = tuple(output_map_shape) + (4, )
            ostart = stride / 2.0
            oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
            oy = oy.reshape(output_shape[0], 1)
            ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
            ox = ox.reshape(1, output_shape[1])
            center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
            center_anchor_map_template[:, :, 0] = oy
            center_anchor_map_template[:, :, 1] = ox
            for anchor_scale in scales:
                for anchor_aspect_ratio in aspect_ratios:
                    center_anchor_map = center_anchor_map_template.copy()
                    center_anchor_map[:, :, 2] = size * anchor_scale / float(anchor_aspect_ratio) ** 0.5
                    center_anchor_map[:, :, 3] = size * anchor_scale * float(anchor_aspect_ratio) ** 0.5

                    edge_anchor_map = np.concatenate(
                        (center_anchor_map[:, :, :2] - center_anchor_map[:, :, 2:4] / 2.0,
                         center_anchor_map[:, :, :2] + center_anchor_map[:, :, 2:4] / 2.0),
                        axis=-1)
                    anchor_area_map = center_anchor_map[:, :, 2] * center_anchor_map[:, :, 3]
                    center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                    edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                    anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))

        return center_anchors, edge_anchors, anchor_areas


def get_ntsnet(backbone,
               aux=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create NTS-Net model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    aux : bool, default False
        Whether to output auxiliary results.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    net = NTSNet(
        backbone=backbone,
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


def ntsnet_cub(pretrained_backbone=False, aux=True, **kwargs):
    """
    NTS-Net model from 'Learning to Navigate for Fine-grained Classification,' https://arxiv.org/abs/1809.00287.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone.final_pool
    return get_ntsnet(backbone=backbone, aux=aux, model_name="ntsnet_cub", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    aux = True
    pretrained = False

    models = [
        ntsnet_cub,
    ]

    for model in models:

        net = model(pretrained=pretrained, aux=aux)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != ntsnet_cub or weight_count == 29033133)
        else:
            assert (model != ntsnet_cub or weight_count == 28623333)

        x = np.zeros((5, 3, 448, 448), np.float32)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == 200))


if __name__ == "__main__":
    _test()
