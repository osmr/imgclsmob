"""
    NTS-Net for CUB-200-2011, implemented in Gluon.
    Original paper: 'Learning to Navigate for Fine-grained Classification,' https://arxiv.org/abs/1809.00287.
"""

__all__ = ['NTSNet', 'ntsnet_cub']

import os
import numpy as np
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv3x3
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


class NavigatorBranch(HybridBlock):
    """
    Navigator branch block for Navigator unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(NavigatorBranch, self).__init__(**kwargs)
        mid_channels = 128

        with self.name_scope():
            self.down_conv = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=strides,
                use_bias=True)
            self.activ = nn.Activation("relu")
            self.tidy_conv = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)
            self.flatten = nn.Flatten()

    def hybrid_forward(self, F, x):
        y = self.down_conv(x)
        y = self.activ(y)
        z = self.tidy_conv(y)
        z = self.flatten(z)
        return z, y


class NavigatorUnit(HybridBlock):
    """
    Navigator init.
    """
    def __init__(self,
                 **kwargs):
        super(NavigatorUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.branch1 = NavigatorBranch(
                in_channels=2048,
                out_channels=6,
                strides=1)
            self.branch2 = NavigatorBranch(
                in_channels=128,
                out_channels=6,
                strides=2)
            self.branch3 = NavigatorBranch(
                in_channels=128,
                out_channels=9,
                strides=2)

    def hybrid_forward(self, F, x):
        t1, x = self.branch1(x)
        t2, x = self.branch2(x)
        t3, _ = self.branch3(x)
        return F.concat(t1, t2, t3, dim=1)


class NTSNet(HybridBlock):
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
                 classes=200,
                 **kwargs):
        super(NTSNet, self).__init__(**kwargs)
        assert (in_channels > 0)
        self.in_size = in_size
        self.classes = classes

        self.top_n = top_n
        self.aux = aux
        self.num_cat = 4
        pad_side = 224
        self.pad_width = (0, 0, 0, 0, pad_side, pad_side, pad_side, pad_side)

        _, edge_anchors, _ = self._generate_default_anchor_maps()
        self.edge_anchors = (edge_anchors + 224).astype(np.int)
        self.edge_anchors = np.concatenate(
            (self.edge_anchors.copy(), np.arange(0, len(self.edge_anchors)).reshape(-1, 1)), axis=1)

        with self.name_scope():
            self.backbone = backbone

            self.backbone_tail = nn.HybridSequential(prefix="")
            self.backbone_tail.add(nn.GlobalAvgPool2D())
            self.backbone_tail.add(nn.Flatten())
            self.backbone_tail.add(nn.Dropout(rate=0.5))
            self.backbone_classifier = nn.Dense(
                units=classes,
                in_units=(512 * 4))

            self.navigator_unit = NavigatorUnit()
            self.concat_net = nn.Dense(
                units=classes,
                in_units=(2048 * (self.num_cat + 1)))

            if self.aux:
                self.partcls_net = nn.Dense(
                    units=classes,
                    in_units=(512 * 4))

    def hybrid_forward(self, F, x):
        raw_pre_features = self.backbone(x)

        rpn_score = self.navigator_unit(raw_pre_features)
        all_cdds = [np.concatenate((y.reshape(-1, 1), self.edge_anchors.copy()), axis=1)
                    for y in rpn_score.asnumpy()]
        top_n_cdds = [hard_nms(y, top_n=self.top_n, iou_thresh=0.25) for y in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int64)
        top_n_index2 = mx.nd.array(np.array([np.repeat(np.arange(top_n_cdds.shape[0]), top_n_cdds.shape[1]),
                                             top_n_index.flatten()]), dtype=np.int64)
        top_n_prob = F.gather_nd(rpn_score, top_n_index2).reshape(x.shape[0], -1)

        batch = x.shape[0]
        part_imgs = mx.nd.zeros(shape=(batch, self.top_n, 3, 224, 224), ctx=x.context, dtype=x.dtype)
        x_pad = F.pad(x, mode="constant", pad_width=self.pad_width, constant_value=0)
        for i in range(batch):
            for j in range(self.top_n):
                y0, x0, y1, x1 = tuple(top_n_cdds[i][j, 1:5].astype(np.int64))
                part_imgs[i:i + 1, j] = F.contrib.BilinearResize2D(
                    x_pad[i:i + 1, :, y0:y1, x0:x1],
                    height=224,
                    width=224)
        part_imgs = part_imgs.reshape((batch * self.top_n, 3, 224, 224))
        part_features = self.backbone_tail(self.backbone(part_imgs.detach()))

        part_feature = part_features.reshape((batch, self.top_n, -1))
        part_feature = part_feature[:, :self.num_cat, :]
        part_feature = part_feature.reshape((batch, -1))

        raw_features = self.backbone_tail(raw_pre_features.detach())

        concat_out = F.concat(part_feature, raw_features, dim=1)
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
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx,
            ignore_extra=True)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features[:-1]
    return get_ntsnet(backbone=backbone, aux=aux, model_name="ntsnet_cub", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    aux = True
    pretrained = False

    models = [
        ntsnet_cub,
    ]

    for model in models:

        net = model(pretrained=pretrained, aux=aux)

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
            assert (model != ntsnet_cub or weight_count == 29033133)
        else:
            assert (model != ntsnet_cub or weight_count == 28623333)

        x = mx.nd.zeros((5, 3, 448, 448), ctx=ctx)
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape[0] == x.shape[0]) and (y.shape[1] == 200)


if __name__ == "__main__":
    _test()
