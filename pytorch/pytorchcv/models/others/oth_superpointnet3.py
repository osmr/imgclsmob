import torch
import torch.nn as nn
import torch.nn.functional as F


class SPConvBlock(nn.Module):
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
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(SPConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


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
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True)


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
        stride=1,
        padding=1)


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
        self.conv1 = sp_conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = sp_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels)

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
        heatmap = heatmap.reshape((-1, x_height, x_height, self.reduction, self.reduction))
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
                if heatmap_mask2_i[pt[1], pt[0]] == 1:
                    heatmap_mask2_i[(pt[1] - pad):(pt[1] + pad + 1), (pt[0] - pad):(pt[0] + pad + 1)] = 0
                    if (bord < pt[0] <= img_width - bord) and (bord < pt[1] <= img_height - bord):
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
    reduction : int, default 8
        Feature reduction factor.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 descriptor_length=256,
                 reduction=8):
        super(SPDescriptor, self).__init__()
        self.desc_length = descriptor_length
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
            pts = pts.unsqueeze(0).unsqueeze(0)
            descriptors = F.grid_sample(coarse_desc_map[i:(i + 1)], pts)
            descriptors = descriptors.squeeze(0).squeeze(1)
            descriptors = descriptors.transpose(0, 1)
            descriptors = F.normalize(descriptors)
            descriptors_list.append(descriptors)

        return descriptors_list


class SuperPointNet(nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()

        in_channels = 1
        channels_per_layers = [64, 64, 128, 128]
        layers = [2, 2, 2, 2]
        channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
        final_block_channels = 256

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if i != 0:
                    stage.add_module("reduce{}".format(i + 1), nn.MaxPool2d(
                        kernel_size=2,
                        stride=2))
                stage.add_module("unit{}".format(j + 1), sp_conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.detector = SPDetector(
            in_channels=in_channels,
            mid_channels=final_block_channels)

        self.descriptor = SPDescriptor(
            in_channels=in_channels,
            mid_channels=final_block_channels)

    def forward(self, x):
        x = self.features(x)
        pts_list, confs_list = self.detector(x)
        descriptors_list = self.descriptor(x, pts_list)
        return pts_list, confs_list, descriptors_list


def oth_superpointnet(pretrained=False, **kwargs):
    return SuperPointNet(**kwargs)


def load_model(net,
               file_path,
               ignore_extra=True):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


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
        oth_superpointnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_superpointnet or weight_count == 1300865)

        x = torch.randn(1, 1, 224, 224)
        y = net(x)
        # y[0][0].sum().backward()
        assert (len(y) == 3)
        # assert (len(y) == 2) and (y[0].shape[0] == y[1].shape[0] == 1) and (y[0].shape[2] == y[1].shape[2] == 28) and\
        #        (y[0].shape[3] == y[1].shape[3] == 28)
        # assert (y[0].shape[1] == 65) and (y[1].shape[1] == 256)


if __name__ == "__main__":
    _test()
