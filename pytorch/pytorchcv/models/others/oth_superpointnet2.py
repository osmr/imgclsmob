import numpy as np
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

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def forward(self, x):
        x_height, x_width = x.size()[-2:]
        img_height = x_height * self.reduction
        img_width = x_width * self.reduction

        semi = self.detector(x)

        dense = semi.softmax(dim=1)
        nodust = dense[:, :-1, :, :]

        heatmap = nodust.permute(0, 2, 3, 1)
        heatmap = heatmap.reshape((-1, x_height, x_height, self.reduction, self.reduction))
        heatmap = heatmap.permute(0, 1, 3, 2, 4)
        heatmap = heatmap.reshape((-1, x_height * self.reduction, x_width * self.reduction))
        heatmap_mask = (heatmap >= self.conf_thresh)
        pts1 = torch.nonzero(heatmap_mask)
        pts2 = torch.masked_select(heatmap, heatmap_mask)

        heatmap = heatmap.data.cpu().numpy().squeeze()
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, img_height, img_width, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_size
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (img_width - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (img_height - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]

        return pts


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

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def forward(self, x, pts):
        x_height, x_width = x.size()[-2:]

        coarse_desc = self.head(x)
        coarse_desc = F.normalize(coarse_desc)
        if pts.shape[1] == 0:
            descriptors = np.zeros((self.desc_length, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            sampled_pts = torch.from_numpy(pts[:2, :].copy())
            sampled_pts[0, :] = (sampled_pts[0, :] / (float(x_height * self.reduction) / 2.)) - 1.
            sampled_pts[1, :] = (sampled_pts[1, :] / (float(x_width * self.reduction) / 2.)) - 1.
            sampled_pts = sampled_pts.transpose(0, 1).contiguous()
            sampled_pts = sampled_pts.view(1, 1, -1, 2)
            sampled_pts = sampled_pts.float()
            sampled_pts = sampled_pts.to(x.device)
            descriptors = F.grid_sample(coarse_desc, sampled_pts)
            descriptors = descriptors.data.cpu().numpy().reshape(self.desc_length, -1)
            descriptors /= np.linalg.norm(descriptors, axis=0)[np.newaxis, :]
        return descriptors


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
        feature_points = self.detector(x)
        descriptors = self.descriptor(x, feature_points)
        return feature_points, descriptors


def oth_superpointnet(pretrained=False, **kwargs):
    return SuperPointNet(**kwargs)


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
        # y.sum().backward()
        assert (len(y) == 2)
        # assert (len(y) == 2) and (y[0].shape[0] == y[1].shape[0] == 1) and (y[0].shape[2] == y[1].shape[2] == 28) and\
        #        (y[0].shape[3] == y[1].shape[3] == 28)
        # assert (y[0].shape[1] == 65) and (y[1].shape[1] == 256)


if __name__ == "__main__":
    _test()
