"""
    SuperPointNet for HPatches (image matching), implemented in Gluon.
    Original paper: 'SuperPoint: Self-Supervised Interest Point Detection and Description,'
    https://arxiv.org/abs/1712.07629.
"""

__all__ = ['SuperPointNet', 'superpointnet']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


def sp_conv1x1(in_channels,
               out_channels):
    """
    1x1 version of the SuperPointNet specific convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        use_bias=True,
        in_channels=in_channels)


class SPConvBlock(HybridBlock):
    """
    SuperPointNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 **kwargs):
        super(SPConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=True,
                in_channels=in_channels)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


def sp_conv3x3_block(in_channels,
                     out_channels):
    """
    3x3 version of the SuperPointNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return SPConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1)


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
            self.conv1 = sp_conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = sp_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels)

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
    border_size : int, default 4
        Image border size to remove points.
    reduction : int, default 8
        Feature reduction factor.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 conf_thresh=0.015,
                 nms_dist=4,
                 border_size=4,
                 reduction=8,
                 **kwargs):
        super(SPDetector, self).__init__(**kwargs)
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist
        self.border_size = border_size
        self.reduction = reduction
        num_classes = reduction * reduction + 1

        with self.name_scope():
            self.detector = SPHead(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=num_classes)

    def hybrid_forward(self, F, x):
        # batch = x.shape[0]
        # x_height, x_width = x.shape[-2:]
        # img_height = x_height * self.reduction
        # img_width = x_width * self.reduction

        semi = self.detector(x)

        dense = semi.softmax(axis=1)
        nodust = dense.slice_axis(axis=1, begin=0, end=-1)

        heatmap = nodust.transpose(axes=(0, 2, 3, 1))
        heatmap = heatmap.reshape(shape=(0, 0, 0, self.reduction, self.reduction))
        heatmap = heatmap.transpose(axes=(0, 1, 3, 2, 4))
        heatmap = heatmap.reshape(shape=(0, -3, -3)).expand_dims(axis=1)
        heatmap = F.where(heatmap >= self.conf_thresh, heatmap, F.zeros_like(heatmap))
        pad = self.nms_dist
        bord = self.border_size + pad
        heatmap_padded = heatmap.pad(mode="constant", pad_width=(0, 0, 0, 0, pad, pad, pad, pad), constant_value=0.0)

        heatmap_mask = (heatmap >= self.conf_thresh)
        heatmap_mask2 = heatmap_mask.pad(mode="constant", pad_width=(0, 0, 0, 0, pad, pad, pad, pad), constant_value=0)
        pts_list = []
        confs_list = []
        # for i in range(batch):
        #     heatmap_i = heatmap[i, 0]
        #     heatmap_mask_i = heatmap_mask[i, 0]
        #     heatmap_mask2_i = heatmap_mask2[i, 0]
        #     src_pts = mxnet.nonzero(heatmap_mask_i)
        #     src_confs = mxnet.masked_select(heatmap_i, heatmap_mask_i)
        #     src_inds = mxnet.argsort(src_confs, descending=True)
        #     dst_inds = mxnet.zeros_like(src_inds)
        #     dst_pts_count = 0
        #     for ind_j in src_inds:
        #         pt = src_pts[ind_j] + pad
        #         assert (pad <= pt[0] < heatmap_mask2_i.shape[0] - pad)
        #         assert (pad <= pt[1] < heatmap_mask2_i.shape[1] - pad)
        #         assert (0 <= pt[0] - pad < img_height)
        #         assert (0 <= pt[1] - pad < img_width)
        #         if heatmap_mask2_i[pt[0], pt[1]] == 1:
        #             heatmap_mask2_i[(pt[0] - pad):(pt[0] + pad + 1), (pt[1] - pad):(pt[1] + pad + 1)] = 0
        #             if (bord < pt[0] - pad <= img_height - bord) and (bord < pt[1] - pad <= img_width - bord):
        #                 dst_inds[dst_pts_count] = ind_j
        #                 dst_pts_count += 1
        #     dst_inds = dst_inds[:dst_pts_count]
        #     dst_pts = mxnet.index_select(src_pts, dim=0, index=dst_inds)
        #     dst_confs = mxnet.index_select(src_confs, dim=0, index=dst_inds)
        #     pts_list.append(dst_pts)
        #     confs_list.append(dst_confs)

        return pts_list, confs_list


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
    reduction : int, default 8
        Feature reduction factor.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 descriptor_length=256,
                 transpose_descriptors=True,
                 reduction=8,
                 **kwargs):
        super(SPDescriptor, self).__init__(**kwargs)
        self.desc_length = descriptor_length
        self.transpose_descriptors = transpose_descriptors
        self.reduction = reduction

        with self.name_scope():
            self.head = SPHead(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=descriptor_length)

    def forward(self, x, pts_list):
        x_height, x_width = x.size()[-2:]

        coarse_desc_map = self.head(x)
        coarse_desc_map = F.normalize(coarse_desc_map)

        descriptors_list = []
        for i, pts in enumerate(pts_list):
            pts = pts.float()
            pts[:, 0] = pts[:, 0] / (0.5 * x_height * self.reduction) - 1.0
            pts[:, 1] = pts[:, 1] / (0.5 * x_width * self.reduction) - 1.0
            if self.transpose_descriptors:
                pts = mxnet.index_select(pts, dim=1, index=mxnet.LongTensor([1, 0]))
            pts = pts.unsqueeze(0).unsqueeze(0)
            descriptors = F.grid_sample(coarse_desc_map[i:(i + 1)], pts)
            descriptors = descriptors.squeeze(0).squeeze(1)
            descriptors = descriptors.transpose(0, 1)
            descriptors = F.normalize(descriptors)
            descriptors_list.append(descriptors)

        return descriptors_list


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
    in_channels : int, default 1
        Number of input channels.
    """
    def __init__(self,
                 channels,
                 final_block_channels,
                 in_channels=1,
                 **kwargs):
        super(SuperPointNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                for j, out_channels in enumerate(channels_per_stage):
                    if (j == 0) and (i != 0):
                        stage.add(nn.MaxPool2D(
                            pool_size=2,
                            strides=2))
                    stage.add(sp_conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels))
                    in_channels = out_channels
                self.features.add(stage)

            self.detector = SPDetector(
                in_channels=in_channels,
                mid_channels=final_block_channels)

            self.descriptor = SPDescriptor(
                in_channels=in_channels,
                mid_channels=final_block_channels)

    def hybrid_forward(self, F, x):
        # assert (x.shape[1] == 1)
        x = self.features(x)
        pts_list, confs_list = self.detector(x)
        descriptors_list = self.descriptor(x, pts_list)
        return pts_list, confs_list, descriptors_list


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

    models = [
        superpointnet,
    ]

    for model in models:

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
        assert (model != superpointnet or weight_count == 1300865)

        x = mx.nd.zeros((1, 1, 224, 224), ctx=ctx)
        y = net(x)
        assert (len(y) == 3)


if __name__ == "__main__":
    _test()
