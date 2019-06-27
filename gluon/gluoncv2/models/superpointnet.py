"""
    SuperPointNet for HPatches (image matching), implemented in Gluon.
    Original paper: 'SuperPoint: Self-Supervised Interest Point Detection and Description,'
    https://arxiv.org/abs/1712.07629.
"""

__all__ = ['SuperPointNet', 'superpointnet']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1
from .vgg import vgg_conv3x3


class SPHead(HybridBlock):
    """
    SuperPointNet head block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 **kwargs):
        super(SPHead, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = vgg_conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=True,
                use_bn=False)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SPDetector(HybridBlock):
    """
    SuperPointNet detector.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    conf_thresh : float, default 0.015
        Confidence threshold.
    nms_dist : int, default 4
        NMS distance.
    batch_size : int, default 1
        Batch size.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    reduction : int, default 8
        Feature reduction factor.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 conf_thresh=0.015,
                 nms_dist=4,
                 batch_size=1,
                 in_size=(224, 224),
                 reduction=8,
                 **kwargs):
        super(SPDetector, self).__init__(**kwargs)
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist
        self.batch_size = batch_size
        self.in_size = in_size
        self.reduction = reduction
        num_classes = reduction * reduction + 1

        with self.name_scope():
            self.detector = SPHead(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=num_classes)

    def hybrid_forward(self, F, x):
        semi = self.detector(x)

        dense = semi.softmax(axis=1)
        nodust = dense.slice_axis(axis=1, begin=0, end=-1)

        heatmap = nodust.transpose(axes=(0, 2, 3, 1))
        heatmap = heatmap.reshape(shape=(0, 0, 0, self.reduction, self.reduction))
        heatmap = heatmap.transpose(axes=(0, 1, 3, 2, 4))
        heatmap = heatmap.reshape(shape=(0, -1))

        in_nms = F.stack(
            heatmap,
            F.arange(self.in_size[0], repeat=self.in_size[1]).tile((self.batch_size, 1)),
            F.arange(self.in_size[0]).tile((self.batch_size, self.in_size[1])),
            F.zeros_like(heatmap) + self.nms_dist,
            F.zeros_like(heatmap) + self.nms_dist,
            axis=2)
        out_nms = F.contrib.box_nms(
            data=in_nms,
            overlap_thresh=1e-3,
            valid_thresh=self.conf_thresh,
            coord_start=1,
            score_index=0,
            id_index=-1,
            force_suppress=False,
            in_format="center",
            out_format="center")
        out_nms = out_nms.slice_axis(axis=2, begin=0, end=3)

        confs = out_nms.slice_axis(axis=2, begin=0, end=1)
        pts = out_nms.slice_axis(axis=2, begin=1, end=3)

        return pts, confs


class SPDescriptor(HybridBlock):
    """
    SuperPointNet descriptor generator.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    descriptor_length : int, default 256
        Descriptor length.
    transpose_descriptors : bool, default True
        Whether transpose descriptors with respect to points.
    select_descriptors : bool, default True
        Whether select descriptors from map.
    batch_size : int, default 1
        Batch size.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    reduction : int, default 8
        Feature reduction factor.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 descriptor_length=256,
                 transpose_descriptors=True,
                 select_descriptors=True,
                 batch_size=1,
                 in_size=(224, 224),
                 reduction=8,
                 **kwargs):
        super(SPDescriptor, self).__init__(**kwargs)
        self.desc_length = descriptor_length
        self.transpose_descriptors = transpose_descriptors
        self.select_descriptors = select_descriptors
        self.batch_size = batch_size
        self.in_size = in_size
        self.reduction = reduction

        with self.name_scope():
            self.head = SPHead(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=descriptor_length)

    def hybrid_forward(self, F, x, pts):
        coarse_desc_map = self.head(x)
        coarse_desc_map = F.L2Normalization(coarse_desc_map, mode="channel")

        desc_map = F.contrib.BilinearResize2D(coarse_desc_map, height=self.in_size[0], width=self.in_size[1])
        desc_map = F.L2Normalization(desc_map, mode="channel")
        if not self.transpose_descriptors:
            desc_map = desc_map.transpose(axes=(0, 1, 3, 2))
        desc_map = desc_map.reshape(shape=(0, 0, -1))
        desc_map = desc_map.transpose(axes=(0, 2, 1))

        if not self.select_descriptors:
            return desc_map

        desc_map_sorted_list = []
        pts_tr = pts.transpose(axes=(0, 2, 1))
        for i in range(self.batch_size):
            desc_map_i = desc_map[i]
            pts_tr_i = pts_tr[i].reshape(shape=(2, -1))
            pts_ravel_i = F.ravel_multi_index(pts_tr_i, shape=(self.in_size[0], self.in_size[1]))
            desc_map_sorted_i = F.take(desc_map_i, pts_ravel_i)
            desc_map_sorted_list.append(desc_map_sorted_i)
        desc_map_sorted = F.stack(*desc_map_sorted_list)

        return desc_map_sorted


class SuperPointNet(HybridBlock):
    """
    SuperPointNet model from 'SuperPoint: Self-Supervised Interest Point Detection and Description,'
    https://arxiv.org/abs/1712.07629.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    final_block_channels : int
        Number of output channels for the final units.
    transpose_descriptors : bool, default True
        Whether transpose descriptors with respect to points.
    select_descriptors : bool, default True
        Whether select descriptors from map.
    postprocess : bool, default True
        Whether fo postprocessing.
    batch_size : int, default 1
        Batch size.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    in_channels : int, default 1
        Number of input channels.
    """
    def __init__(self,
                 channels,
                 final_block_channels,
                 transpose_descriptors=True,
                 select_descriptors=True,
                 postprocess=True,
                 batch_size=1,
                 in_size=(224, 224),
                 in_channels=1,
                 **kwargs):
        super(SuperPointNet, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.in_size = in_size
        self.postprocess = postprocess

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                for j, out_channels in enumerate(channels_per_stage):
                    if (j == 0) and (i != 0):
                        stage.add(nn.MaxPool2D(
                            pool_size=2,
                            strides=2))
                    stage.add(vgg_conv3x3(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_bias=True,
                        use_bn=False))
                    in_channels = out_channels
                self.features.add(stage)

            self.detector = SPDetector(
                in_channels=in_channels,
                mid_channels=final_block_channels,
                batch_size=batch_size,
                in_size=in_size)

            self.descriptor = SPDescriptor(
                in_channels=in_channels,
                mid_channels=final_block_channels,
                transpose_descriptors=transpose_descriptors,
                select_descriptors=select_descriptors,
                batch_size=batch_size,
                in_size=in_size)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        pts, confs = self.detector(x)
        desc_map = self.descriptor(x, pts)

        if self.postprocess:
            counts = (confs > 0).sum(axis=1)

            pts_list = []
            confs_list = []
            desc_map_list = []

            def slice_array(data, state):
                data_list = state[0]
                ii = state[1]
                count_i = counts[int(ii.asscalar())]
                data_i = data.slice_axis(axis=0, begin=0, end=int(count_i.asscalar()))
                data_list.append(data_i)
                return data, [data_list, ii + 1]

            F.contrib.foreach(slice_array, pts, [pts_list, F.zeros(1)])
            F.contrib.foreach(slice_array, confs, [confs_list, F.zeros(1)])
            F.contrib.foreach(slice_array, desc_map, [desc_map_list, F.zeros(1)])

            return pts_list, confs_list, desc_map_list

        return pts, confs, desc_map


def get_superpointnet(model_name=None,
                      pretrained=False,
                      ctx=cpu(),
                      root=os.path.join("~", ".mxnet", "models"),
                      **kwargs):
    """
    Create SuperPointNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    channels_per_layers = [64, 64, 128, 128]
    layers = [2, 2, 2, 2]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    final_block_channels = 256

    net = SuperPointNet(
        channels=channels,
        final_block_channels=final_block_channels,
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


def superpointnet(**kwargs):
    """
    SuperPointNet model from 'SuperPoint: Self-Supervised Interest Point Detection and Description,'
    https://arxiv.org/abs/1712.07629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_superpointnet(model_name="superpointnet", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False
    batch_size = 1
    in_size = (224, 224)
    postprocess = True

    models = [
        superpointnet,
    ]

    for model in models:

        net = model(pretrained=pretrained, batch_size=batch_size, in_size=in_size, postprocess=postprocess)

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
        assert (model != superpointnet or weight_count == 1300865)

        x = mx.nd.random.normal(shape=(batch_size, 1, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (len(y) == 3)


if __name__ == "__main__":
    _test()
