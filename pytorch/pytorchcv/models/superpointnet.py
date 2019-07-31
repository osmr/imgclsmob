"""
    SuperPointNet for HPatches (image matching), implemented in PyTorch.
    Original paper: 'SuperPoint: Self-Supervised Interest Point Detection and Description,'
    https://arxiv.org/abs/1712.07629.
"""

__all__ = ['SuperPointNet', 'superpointnet']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .common import conv1x1, conv3x3_block


class SPHead(nn.Module):
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
                 out_channels):
        super(SPHead, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            use_bn=False)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SPDetector(nn.Module):
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
                 reduction=8):
        super(SPDetector, self).__init__()
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist
        self.border_size = border_size
        self.reduction = reduction
        num_classes = reduction * reduction + 1

        self.detector = SPHead(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=num_classes)

    def forward(self, x):
        batch = x.size(0)
        x_height, x_width = x.size()[-2:]
        img_height = x_height * self.reduction
        img_width = x_width * self.reduction

        semi = self.detector(x)

        dense = semi.softmax(dim=1)
        nodust = dense[:, :-1, :, :]

        heatmap = nodust.permute(0, 2, 3, 1)
        heatmap = heatmap.reshape((-1, x_height, x_width, self.reduction, self.reduction))
        heatmap = heatmap.permute(0, 1, 3, 2, 4)
        heatmap = heatmap.reshape((-1, 1, x_height * self.reduction, x_width * self.reduction))
        heatmap_mask = (heatmap >= self.conf_thresh)
        pad = self.nms_dist
        bord = self.border_size + pad
        heatmap_mask2 = F.pad(heatmap_mask, pad=(pad, pad, pad, pad))
        pts_list = []
        confs_list = []
        for i in range(batch):
            heatmap_i = heatmap[i, 0]
            heatmap_mask_i = heatmap_mask[i, 0]
            heatmap_mask2_i = heatmap_mask2[i, 0]
            src_pts = torch.nonzero(heatmap_mask_i)
            src_confs = torch.masked_select(heatmap_i, heatmap_mask_i)
            src_inds = torch.argsort(src_confs, descending=True)
            dst_inds = torch.zeros_like(src_inds)
            dst_pts_count = 0
            for ind_j in src_inds:
                pt = src_pts[ind_j] + pad
                assert (pad <= pt[0] < heatmap_mask2_i.shape[0] - pad)
                assert (pad <= pt[1] < heatmap_mask2_i.shape[1] - pad)
                assert (0 <= pt[0] - pad < img_height)
                assert (0 <= pt[1] - pad < img_width)
                if heatmap_mask2_i[pt[0], pt[1]] == 1:
                    heatmap_mask2_i[(pt[0] - pad):(pt[0] + pad + 1), (pt[1] - pad):(pt[1] + pad + 1)] = 0
                    if (bord < pt[0] - pad <= img_height - bord) and (bord < pt[1] - pad <= img_width - bord):
                        dst_inds[dst_pts_count] = ind_j
                        dst_pts_count += 1
            dst_inds = dst_inds[:dst_pts_count]
            dst_pts = torch.index_select(src_pts, dim=0, index=dst_inds)
            dst_confs = torch.index_select(src_confs, dim=0, index=dst_inds)
            pts_list.append(dst_pts)
            confs_list.append(dst_confs)

        return pts_list, confs_list


class SPDescriptor(nn.Module):
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
                 reduction=8):
        super(SPDescriptor, self).__init__()
        self.desc_length = descriptor_length
        self.transpose_descriptors = transpose_descriptors
        self.reduction = reduction

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
                pts = torch.index_select(pts, dim=1, index=torch.tensor([1, 0], device=pts.device))
            pts = pts.unsqueeze(0).unsqueeze(0)
            descriptors = F.grid_sample(coarse_desc_map[i:(i + 1)], pts)
            descriptors = descriptors.squeeze(0).squeeze(1)
            descriptors = descriptors.transpose(0, 1)
            descriptors = F.normalize(descriptors)
            descriptors_list.append(descriptors)

        return descriptors_list


class SuperPointNet(nn.Module):
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
                 transpose_descriptors=True,
                 in_channels=1):
        super(SuperPointNet, self).__init__()
        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    stage.add_module("reduce{}".format(i + 1), nn.MaxPool2d(
                        kernel_size=2,
                        stride=2))
                stage.add_module("unit{}".format(j + 1), conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=True,
                    use_bn=False))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.detector = SPDetector(
            in_channels=in_channels,
            mid_channels=final_block_channels)

        self.descriptor = SPDescriptor(
            in_channels=in_channels,
            mid_channels=final_block_channels,
            transpose_descriptors=transpose_descriptors)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        assert (x.size(1) == 1)
        x = self.features(x)
        pts_list, confs_list = self.detector(x)
        descriptors_list = self.descriptor(x, pts_list)
        return pts_list, confs_list, descriptors_list


def get_superpointnet(model_name=None,
                      pretrained=False,
                      root=os.path.join("~", ".torch", "models"),
                      **kwargs):
    """
    Create SuperPointNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
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
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def superpointnet(**kwargs):
    """
    SuperPointNet model from 'SuperPoint: Self-Supervised Interest Point Detection and Description,'
    https://arxiv.org/abs/1712.07629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_superpointnet(model_name="superpointnet", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        superpointnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != superpointnet or weight_count == 1300865)

        # x = torch.randn(1, 1, 224, 224)
        x = torch.randn(1, 1, 1000, 2000)
        y = net(x)
        # y.sum().backward()
        assert (len(y) == 3)


if __name__ == "__main__":
    _test()
