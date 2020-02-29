"""
    Lightweight OpenPose 3D for CMU Panoptic, implemented in PyTorch.
    Original paper: 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.
"""

__all__ = ['LwOpenPose', 'lwopenpose3d_mobilenet_cmupan']

import os
import math
import numpy as np
from operator import itemgetter
import torch
from torch import nn
from .common import conv1x1, conv1x1_block, conv3x3_block, dwsconv3x3_block, InterpolationBlock


class Heatmap2dMaxDetBlock(nn.Module):
    """
    Heatmap 2D maximum detector block (for 2D human pose estimation task).
    """
    def __init__(self):
        super(Heatmap2dMaxDetBlock, self).__init__()

    def forward(self, heatmap2d, paf2d):
        batch = heatmap2d.shape[0]
        keypoints = heatmap2d.shape[1]

        pts = []
        for batch_i in range(batch):
            heatmap = np.transpose(heatmap2d[batch_i].cpu().data.numpy(), (1, 2, 0))
            paf = np.transpose(paf2d[batch_i].cpu().data.numpy(), (1, 2, 0))
            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(keypoints):
                total_keypoints_num += self.extract_keypoints(
                    heatmap=heatmap[:, :, kpt_idx],
                    all_keypoints=all_keypoints_by_type,
                    total_keypoint_num=total_keypoints_num)

            pose_entries, all_keypoints = self.group_keypoints(
                all_keypoints_by_type=all_keypoints_by_type,
                pafs=paf)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = all_keypoints[kpt_id, 0] * 2.0
                all_keypoints[kpt_id, 1] = all_keypoints[kpt_id, 1] * 2.0
            poses_batch_i = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(keypoints):
                    if pose_entries[n][kpt_id] != -1.0:
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = (pose_keypoints, pose_entries[n][18])
                poses_batch_i.append(pose)
            pts.append(poses_batch_i)
        return pts

    @staticmethod
    def extract_keypoints(heatmap,
                          all_keypoints,
                          total_keypoint_num):
        heatmap[heatmap < 0.1] = 0
        heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
        heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 1:heatmap_with_borders.shape[1] - 1]
        heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 2:heatmap_with_borders.shape[1]]
        heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 0:heatmap_with_borders.shape[1] - 2]
        heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1] - 1]
        heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0] - 2, 1:heatmap_with_borders.shape[1] - 1]

        heatmap_peaks = (heatmap_center > heatmap_left) &\
                        (heatmap_center > heatmap_right) &\
                        (heatmap_center > heatmap_up) &\
                        (heatmap_center > heatmap_down)
        heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0] - 1, 1:heatmap_center.shape[1] - 1]
        keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
        keypoints = sorted(keypoints, key=itemgetter(0))

        suppressed = np.zeros(len(keypoints), np.uint8)
        keypoints_with_score_and_id = []
        keypoint_num = 0
        for i in range(len(keypoints)):
            if suppressed[i]:
                continue
            for j in range(i + 1, len(keypoints)):
                if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 + (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                    suppressed[j] = 1
            keypoint_with_score_and_id = (
                keypoints[i][0],
                keypoints[i][1],
                heatmap[keypoints[i][1], keypoints[i][0]],
                total_keypoint_num + keypoint_num)
            keypoints_with_score_and_id.append(keypoint_with_score_and_id)
            keypoint_num += 1
        all_keypoints.append(keypoints_with_score_and_id)
        return keypoint_num

    @staticmethod
    def group_keypoints(all_keypoints_by_type,
                        pafs,
                        pose_entry_size=20,
                        min_paf_score=0.05):

        def linspace2d(start, stop, n=10):
            points = 1 / (n - 1) * (stop - start)
            return points[:, None] * np.arange(n) + start[:, None]

        BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                              [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
        BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                              [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19],
                              [26, 27])

        pose_entries = []
        all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
        for part_id in range(len(BODY_PARTS_PAF_IDS)):
            part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
            kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
            kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
            num_kpts_a = len(kpts_a)
            num_kpts_b = len(kpts_b)
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
                continue
            elif num_kpts_a == 0:  # body part has just 'b' keypoints
                for i in range(num_kpts_b):
                    num = 0
                    for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                        if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                        pose_entry[-1] = 1                   # num keypoints in pose
                        pose_entry[-2] = kpts_b[i][2]        # pose score
                        pose_entries.append(pose_entry)
                continue
            elif num_kpts_b == 0:  # body part has just 'a' keypoints
                for i in range(num_kpts_a):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = kpts_a[i][3]
                        pose_entry[-1] = 1
                        pose_entry[-2] = kpts_a[i][2]
                        pose_entries.append(pose_entry)
                continue

            connections = []
            for i in range(num_kpts_a):
                kpt_a = np.array(kpts_a[i][0:2])
                for j in range(num_kpts_b):
                    kpt_b = np.array(kpts_b[j][0:2])
                    mid_point = [(), ()]
                    mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                    int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                    mid_point[1] = mid_point[0]

                    vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                    vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                    if vec_norm == 0:
                        continue
                    vec[0] /= vec_norm
                    vec[1] /= vec_norm
                    cur_point_score = (vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0] +
                                       vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1])

                    height_n = pafs.shape[0] // 2
                    success_ratio = 0
                    point_num = 10  # number of points to integration over paf
                    if cur_point_score > -100:
                        passed_point_score = 0
                        passed_point_num = 0
                        x, y = linspace2d(kpt_a, kpt_b)
                        for point_idx in range(point_num):
                            px = int(round(x[point_idx]))
                            py = int(round(y[point_idx]))
                            paf = part_pafs[py, px, 0:2]
                            cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                            if cur_point_score > min_paf_score:
                                passed_point_score += cur_point_score
                                passed_point_num += 1
                        success_ratio = passed_point_num / point_num
                        ratio = 0
                        if passed_point_num > 0:
                            ratio = passed_point_score / passed_point_num
                        ratio += min(height_n / vec_norm - 1, 0)
                    if ratio > 0 and success_ratio > 0.8:
                        score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                        connections.append([i, j, ratio, score_all])
            if len(connections) > 0:
                connections = sorted(connections, key=itemgetter(2), reverse=True)

            num_connections = min(num_kpts_a, num_kpts_b)
            has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
            has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
            filtered_connections = []
            for row in range(len(connections)):
                if len(filtered_connections) == num_connections:
                    break
                i, j, cur_point_score = connections[row][0:3]
                if not has_kpt_a[i] and not has_kpt_b[j]:
                    filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                    has_kpt_a[i] = 1
                    has_kpt_b[j] = 1
            connections = filtered_connections
            if len(connections) == 0:
                continue

            if part_id == 0:
                pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
                for i in range(len(connections)):
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                    pose_entries[i][-1] = 2
                    pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
            elif part_id == 17 or part_id == 18:
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                        elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                            pose_entries[j][kpt_a_id] = connections[i][0]
                continue
            else:
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0]:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                            num += 1
                            pose_entries[j][-1] += 1
                            pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = connections[i][0]
                        pose_entry[kpt_b_id] = connections[i][1]
                        pose_entry[-1] = 2
                        pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                        pose_entries.append(pose_entry)

        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
                continue
            filtered_entries.append(pose_entries[i])
        pose_entries = np.asarray(filtered_entries)
        return pose_entries, all_keypoints


class LwopResBottleneck(nn.Module):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False):
        super(LwopResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor if squeeze_out else in_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            bias=bias)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LwopResUnit(nn.Module):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    activate : bool, default False
        Whether to activate the sum.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False,
                 activate=False):
        super(LwopResUnit, self).__init__()
        self.activate = activate
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = LwopResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            bottleneck_factor=bottleneck_factor,
            squeeze_out=squeeze_out)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                activation=None)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        if self.activate:
            x = self.activ(x)
        return x


class LwopEncoderFinalBlock(nn.Module):
    """
    Lightweight OpenPose 3D specific encoder final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(LwopEncoderFinalBlock, self).__init__()
        self.pre_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)
        self.body = nn.Sequential()
        for i in range(3):
            self.body.add_module("block{}".format(i + 1), dwsconv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bn=False,
                dw_activation=(lambda: nn.ELU(inplace=True)),
                pw_activation=(lambda: nn.ELU(inplace=True))))
        self.post_conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)

    def forward(self, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        x = self.post_conv(x)
        return x


class LwopRefinementBlock(nn.Module):
    """
    Lightweight OpenPose 3D specific refinement block for decoder units.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(LwopRefinementBlock, self).__init__()
        self.pre_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_bn=False)
        self.body = nn.Sequential()
        self.body.add_module("block1", conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=True))
        self.body.add_module("block2", conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=2,
            dilation=2,
            bias=True))

    def forward(self, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        return x


class LwopDecoderBend(nn.Module):
    """
    Lightweight OpenPose 3D specific decoder bend block.

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
        super(LwopDecoderBend, self).__init__()
        self.conv1 = conv1x1_block(
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


class LwopDecoderInitBlock(nn.Module):
    """
    Lightweight OpenPose 3D specific decoder init block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    num_heatmap : int
        Number of heatmaps.
    num_paf : int
        Number of PAFs.
    """
    def __init__(self,
                 in_channels,
                 num_heatmap,
                 num_paf):
        super(LwopDecoderInitBlock, self).__init__()
        bend_mid_channels = 512

        self.body = nn.Sequential()
        for i in range(3):
            self.body.add_module("block{}".format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=True,
                use_bn=False))
        self.heatmap_bend = LwopDecoderBend(
            in_channels=in_channels,
            mid_channels=bend_mid_channels,
            out_channels=num_heatmap)
        self.paf_bend = LwopDecoderBend(
            in_channels=in_channels,
            mid_channels=bend_mid_channels,
            out_channels=num_paf)

    def forward(self, x):
        x = self.body(x)
        heatmap = self.heatmap_bend(x)
        paf = self.paf_bend(x)
        y = torch.cat((x, heatmap, paf), dim=1)
        return y


class LwopDecoderUnit(nn.Module):
    """
    Lightweight OpenPose 3D specific decoder init.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    num_heatmap : int
        Number of heatmaps.
    num_paf : int
        Number of PAFs.
    """
    def __init__(self,
                 in_channels,
                 num_heatmap,
                 num_paf):
        super(LwopDecoderUnit, self).__init__()
        features_channels = in_channels - num_heatmap - num_paf

        self.body = nn.Sequential()
        for i in range(5):
            self.body.add_module("block{}".format(i + 1), LwopRefinementBlock(
                in_channels=in_channels,
                out_channels=features_channels))
            in_channels = features_channels
        self.heatmap_bend = LwopDecoderBend(
            in_channels=features_channels,
            mid_channels=features_channels,
            out_channels=num_heatmap)
        self.paf_bend = LwopDecoderBend(
            in_channels=features_channels,
            mid_channels=features_channels,
            out_channels=num_paf)

    def forward(self, x):
        x = self.body(x)
        heatmap = self.heatmap_bend(x)
        paf = self.paf_bend(x)
        y = torch.cat((x, heatmap, paf), dim=1)
        return y


class LwopDecoderFeaturesBend(nn.Module):
    """
    Lightweight OpenPose 3D specific decoder 3D features bend.

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
        super(LwopDecoderFeaturesBend, self).__init__()
        self.body = nn.Sequential()
        for i in range(2):
            self.body.add_module("block{}".format(i + 1), LwopRefinementBlock(
                in_channels=in_channels,
                out_channels=mid_channels))
            in_channels = mid_channels
        self.features_bend = LwopDecoderBend(
            in_channels=mid_channels,
            mid_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.body(x)
        x = self.features_bend(x)
        return x


class LwopDecoderFinalBlock(nn.Module):
    """
    Lightweight OpenPose 3D specific decoder final block for calcualation 3D poses.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    num_heatmap : int
        Number of heatmaps.
    num_paf : int
        Number of PAFs.
    bottleneck_factor : int
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 num_heatmap,
                 num_paf,
                 bottleneck_factor):
        super(LwopDecoderFinalBlock, self).__init__()
        self.num_heatmap = num_heatmap
        self.num_paf = num_paf
        features_out_channels = num_heatmap + num_paf
        features_in_channels = in_channels - features_out_channels

        self.body = nn.Sequential()
        for i in range(5):
            self.body.add_module("block{}".format(i + 1), LwopResUnit(
                in_channels=in_channels,
                out_channels=features_in_channels,
                bottleneck_factor=bottleneck_factor))
            in_channels = features_in_channels
        self.features_bend = LwopDecoderFeaturesBend(
            in_channels=features_in_channels,
            mid_channels=features_in_channels,
            out_channels=features_out_channels)

    def forward(self, x):
        heatmap_paf_2d = x[:, -self.num_paf - self.num_heatmap:]
        x = self.body(x)
        x = self.features_bend(x)
        y = torch.cat((x, heatmap_paf_2d), dim=1)
        return y


class LwOpenPose(nn.Module):
    """
    Lightweight OpenPose 3D model from 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    encoder_channels : list of list of int
        Number of output channels for each encoder unit.
    encoder_paddings : list of list of int
        Padding/dilation value for each encoder unit.
    encoder_init_block_channels : int
        Number of output channels for the encoder initial unit.
    encoder_final_block_channels : int
        Number of output channels for the encoder final unit.
    refinement_units : int
        Number of refinement blocks in the decoder.
    return_heatmap : bool, default False
        Whether to return only heatmap.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 192)
        Spatial size of the expected input image.
    keypoints : int, default 19
        Number of keypoints.
    """
    def __init__(self,
                 encoder_channels,
                 encoder_paddings,
                 encoder_init_block_channels,
                 encoder_final_block_channels,
                 refinement_units,
                 return_heatmap=False,
                 in_channels=3,
                 in_size=(256, 256),
                 keypoints=19):
        super(LwOpenPose, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap
        num_heatmap = keypoints
        num_paf = 2 * keypoints

        self.encoder = nn.Sequential()
        backbone = nn.Sequential()
        backbone.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=encoder_init_block_channels,
            stride=2))
        in_channels = encoder_init_block_channels
        for i, channels_per_stage in enumerate(encoder_channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                padding = encoder_paddings[i][j]
                stage.add_module("unit{}".format(j + 1), dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    dilation=padding))
                in_channels = out_channels
            backbone.add_module("stage{}".format(i + 1), stage)
        self.encoder.add_module("backbone", backbone)
        self.encoder.add_module("final_block", LwopEncoderFinalBlock(
            in_channels=in_channels,
            out_channels=encoder_final_block_channels))
        in_channels = encoder_final_block_channels

        self.decoder = nn.Sequential()
        self.decoder.add_module("init_block", LwopDecoderInitBlock(
            in_channels=in_channels,
            num_heatmap=num_heatmap,
            num_paf=num_paf))
        in_channels = encoder_final_block_channels + num_heatmap + num_paf
        for i in range(refinement_units):
            self.decoder.add_module("unit{}".format(i + 1), LwopDecoderUnit(
                in_channels=in_channels,
                num_heatmap=num_heatmap,
                num_paf=num_paf))
        self.decoder.add_module("final_block", LwopDecoderFinalBlock(
            in_channels=in_channels,
            num_heatmap=num_heatmap,
            num_paf=num_paf,
            bottleneck_factor=2))

        self.up = InterpolationBlock(scale_factor=4)
        self.heatmap_max_det_2d = Heatmap2dMaxDetBlock()

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        heatmap_paf = self.decoder(x)
        if self.return_heatmap:
            return heatmap_paf
        else:
            num_heatmap = self.keypoints
            num_paf = 2 * self.keypoints
            heatmap2d = x[:, -num_paf - num_heatmap:-num_paf]
            paf2d = x[:, -num_paf:]
            heatmap2d = self.up(heatmap2d)
            paf2d = self.up(paf2d)
            pts = self.heatmap_max_det_2d(heatmap2d, paf2d)
            return pts


def get_lwopenpose3d(keypoints,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".torch", "models"),
                     **kwargs):
    """
    Create Lightweight OpenPose 3D model with specific parameters.

    Parameters:
    ----------
    keypoints : int
        Number of keypoints.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    encoder_channels = [[64], [128, 128], [256, 256, 512, 512, 512, 512, 512, 512]]
    encoder_paddings = [[1], [1, 1], [1, 1, 1, 2, 1, 1, 1, 1]]
    encoder_init_block_channels = 32
    encoder_final_block_channels = 128
    refinement_units = 1

    net = LwOpenPose(
        encoder_channels=encoder_channels,
        encoder_paddings=encoder_paddings,
        encoder_init_block_channels=encoder_init_block_channels,
        encoder_final_block_channels=encoder_final_block_channels,
        refinement_units=refinement_units,
        keypoints=keypoints,
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


def lwopenpose3d_mobilenet_cmupan(keypoints=19, **kwargs):
    """
    Lightweight OpenPose 3D model on the base of MobileNet for CMU Panoptic from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose3d(keypoints=keypoints, model_name="lwopenpose3d_mobilenet_cmupan", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    in_size = (256, 256)
    keypoints = 19
    return_heatmap = True
    pretrained = False

    models = [
        lwopenpose3d_mobilenet_cmupan,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lwopenpose3d_mobilenet_cmupan or weight_count == 5085983)

        batch = 1
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 6 * keypoints, in_size[0] // 8, in_size[0] // 8))


if __name__ == "__main__":
    _test()
