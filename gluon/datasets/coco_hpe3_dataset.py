"""
    COCO keypoint detection (2D multiple human pose estimation) dataset (for IBPPose).
"""

import os
# import json
import math
import cv2
import numpy as np
from mxnet.gluon.data import dataset
from .dataset_metainfo import DatasetMetaInfo


class CocoHpe3Dataset(dataset.Dataset):
    """
    COCO keypoint detection (2D multiple human pose estimation) dataset.

    Parameters
    ----------
    root : string
        Path to `annotations`, `train2017`, and `val2017` folders.
    mode : string, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode="train",
                 transform=None):
        super(CocoHpe3Dataset, self).__init__()
        self._root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform

        mode_name = "train" if mode == "train" else "val"
        annotations_dir_path = os.path.join(root, "annotations")
        annotations_file_path = os.path.join(annotations_dir_path, "person_keypoints_" + mode_name + "2017.json")
        # with open(annotations_file_path, "r") as f:
        #     self.file_names = json.load(f)["images"]
        self.image_dir_path = os.path.join(root, mode_name + "2017")
        self.annotations_file_path = annotations_file_path

        from pycocotools.coco import COCO
        self.coco_gt = COCO(self.annotations_file_path)
        self.validation_ids = self.coco_gt.getImgIds()[:]

    def __str__(self):
        return self.__class__.__name__ + "(" + self._root + ")"

    def __len__(self):
        return len(self.validation_ids)

    def __getitem__(self, idx):
        # file_name = self.file_names[idx]["file_name"]
        image_id = self.validation_ids[idx]
        file_name = self.coco_gt.imgs[image_id]["file_name"]
        image_file_path = os.path.join(self.image_dir_path, file_name)
        image = cv2.imread(image_file_path, flags=cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        image_src_shape = image.shape[:2]

        boxsize = 512
        max_downsample = 64
        pad_value = 128
        scale = boxsize / image.shape[0]
        if scale * image.shape[0] > 2600 or scale * image.shape[1] > 3800:
            scale = min(2600 / image.shape[0], 3800 / image.shape[1])
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        image, pad = self.pad_right_down_corner(image, max_downsample, pad_value)
        image = np.float32(image / 255)
        image = image.transpose((2, 0, 1))

        # image_id = int(os.path.splitext(os.path.basename(file_name))[0])

        label = np.array([image_id, 1.0] + pad + list(image_src_shape), np.float32)

        return image, label

    @staticmethod
    def pad_right_down_corner(img,
                              stride,
                              pad_value):
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
        pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        return img_padded, pad


# ---------------------------------------------------------------------------------------------------------------------


class CocoHpe2ValTransform(object):
    def __init__(self,
                 ds_metainfo):
        self.ds_metainfo = ds_metainfo

    def __call__(self, src, label):
        return src, label


def recalc_pose(pred,
                label):
    dt_gt_mapping = {0: 0, 1: None, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
                     14: 2, 15: 1, 16: 4, 17: 3}
    parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne",
             "Lank", "Reye", "Leye", "Rear", "Lear"]
    num_parts = len(parts)
    parts_dict = dict(zip(parts, range(num_parts)))
    limb_from = ['neck', 'neck', 'neck', 'neck', 'neck', 'nose', 'nose', 'Reye', 'Leye', 'neck', 'Rsho', 'Relb', 'neck',
                 'Lsho', 'Lelb', 'neck', 'Rhip', 'Rkne', 'neck', 'Lhip', 'Lkne', 'nose', 'nose', 'Rsho', 'Rhip', 'Lsho',
                 'Lhip', 'Rear', 'Lear', 'Rhip']
    limb_to = ['nose', 'Reye', 'Leye', 'Rear', 'Lear', 'Reye', 'Leye', 'Rear', 'Lear', 'Rsho', 'Relb', 'Rwri', 'Lsho',
               'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Rsho', 'Lsho', 'Rhip', 'Lkne', 'Lhip',
               'Rkne', 'Rsho', 'Lsho', 'Lhip']
    limb_from = [parts_dict[n] for n in limb_from]
    limb_to = [parts_dict[n] for n in limb_to]
    assert limb_from == [x for x in [
        1, 1, 1, 1, 1, 0, 0, 14, 15, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 2, 8, 5, 11, 16, 17, 8]]
    assert limb_to == [x for x in [
        0, 14, 15, 16, 17, 14, 15, 16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 5, 8, 12, 11, 9, 2, 5, 11]]
    limbs_conn = list(zip(limb_from, limb_to))
    limb_seq = limbs_conn

    paf_layers = 30
    num_layers = 50
    stride = 4

    label_img_id = label[:, 0].astype(np.int32)
    # label_score = label[:, 1]

    pads = label[:, 2:6].astype(np.int32)
    image_src_shapes = label[:, 6:8].astype(np.int32)

    pred_pts_score = []
    pred_person_score = []
    label_img_id_ = []

    batch = pred.shape[0]
    for batch_i in range(batch):
        label_img_id_i = label_img_id[batch_i]
        pad = list(pads[batch_i])
        image_src_shape = list(image_src_shapes[batch_i])

        output_blob = pred[batch_i].transpose((1, 2, 0))
        output_paf = output_blob[:, :, :paf_layers]
        output_heatmap = output_blob[:, :, paf_layers:num_layers]

        heatmap = cv2.resize(output_heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[
                  pad[0]:(output_blob.shape[0] * stride - pad[2]),
                  pad[1]:(output_blob.shape[1] * stride - pad[3]),
                  :]
        heatmap = cv2.resize(heatmap, (image_src_shape[1], image_src_shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = cv2.resize(output_paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[
              pad[0]:(output_blob.shape[0] * stride - pad[2]),
              pad[1]:(output_blob.shape[1] * stride - pad[3]),
              :]
        paf = cv2.resize(paf, (image_src_shape[1], image_src_shape[0]), interpolation=cv2.INTER_CUBIC)

        all_peaks = find_peaks(heatmap)
        connection_all, special_k = find_connections(all_peaks, paf, image_src_shape[0], limb_seq)
        subset, candidate = find_people(connection_all, special_k, all_peaks, limb_seq)

        for s in subset[..., 0]:
            keypoint_indexes = s[:18]
            person_keypoint_coordinates = []
            for index in keypoint_indexes:
                if index == -1:
                    X, Y, C = 0, 0, 0
                else:
                    X, Y, C = list(candidate[index.astype(int)][:2]) + [1]
                person_keypoint_coordinates.append([X, Y, C])
            person_keypoint_coordinates_coco = [None] * 17

            for dt_index, gt_index in dt_gt_mapping.items():
                if gt_index is None:
                    continue
                person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

            pred_pts_score.append(person_keypoint_coordinates_coco)
            pred_person_score.append(1 - 1.0 / s[18])
            label_img_id_.append(label_img_id_i)

    return np.array(pred_pts_score).reshape((-1, 17, 3)), np.array(pred_person_score), np.array(label_img_id_)


def find_peaks(heatmap_avg):
    import torch

    thre1 = 0.1
    offset_radius = 2

    all_peaks = []
    peak_counter = 0

    heatmap_avg = heatmap_avg.astype(np.float32)

    filter_map = heatmap_avg[:, :, :18].copy().transpose((2, 0, 1))[None, ...]
    filter_map = torch.from_numpy(filter_map).cuda()

    filter_map = keypoint_heatmap_nms(filter_map, kernel=3, thre=thre1)
    filter_map = filter_map.cpu().numpy().squeeze().transpose((1, 2, 0))

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        peaks_binary = filter_map[:, :, part]
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        refined_peaks_with_score = [refine_centroid(map_ori, anchor, offset_radius) for anchor in peaks]

        id = range(peak_counter, peak_counter + len(refined_peaks_with_score))
        peaks_with_score_and_id = [refined_peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    return all_peaks


def keypoint_heatmap_nms(heat, kernel=3, thre=0.1):
    from torch.nn import functional as F

    # keypoint NMS on heatmap (score map)
    pad = (kernel - 1) // 2
    pad_heat = F.pad(heat, (pad, pad, pad, pad), mode="reflect")
    hmax = F.max_pool2d(pad_heat, (kernel, kernel), stride=1, padding=0)
    keep = (hmax == heat).float() * (heat >= thre).float()
    return heat * keep


def refine_centroid(scorefmp, anchor, radius):
    """
    Refine the centroid coordinate. It dose not affect the results after testing.
    :param scorefmp: 2-D numpy array, original regressed score map
    :param anchor: python tuple, (x,y) coordinates
    :param radius: int, range of considered scores
    :return: refined anchor, refined score
    """

    x_c, y_c = anchor
    x_min = x_c - radius
    x_max = x_c + radius + 1
    y_min = y_c - radius
    y_max = y_c + radius + 1

    if y_max > scorefmp.shape[0] or y_min < 0 or x_max > scorefmp.shape[1] or x_min < 0:
        return anchor + (scorefmp[y_c, x_c], )

    score_box = scorefmp[y_min:y_max, x_min:x_max]
    x_grid, y_grid = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    offset_x = (score_box * x_grid).sum() / score_box.sum()
    offset_y = (score_box * y_grid).sum() / score_box.sum()
    x_refine = x_c + offset_x
    y_refine = y_c + offset_y
    refined_anchor = (x_refine, y_refine)
    return refined_anchor + (score_box.mean(),)


def find_connections(all_peaks, paf_avg, image_width, limb_seq):
    mid_num_ = 20
    thre2 = 0.1
    connect_ration = 0.8

    connection_all = []
    special_k = []

    for k in range(len(limb_seq)):
        score_mid = paf_avg[:, :, k]
        candA = all_peaks[limb_seq[k][0]]
        candB = all_peaks[limb_seq[k][1]]
        nA = len(candA)
        nB = len(candB)
        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    mid_num = min(int(round(norm + 1)), mid_num_)
                    if norm == 0:
                        continue

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    limb_response = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0]))] for
                                              I in range(len(startend))])

                    score_midpts = limb_response

                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * image_width / norm - 1, 0)

                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) >= connect_ration * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([
                            i,
                            j,
                            score_with_dist_prior,
                            norm,
                            0.5 * score_with_dist_prior + 0.25 * candA[i][2] + 0.25 * candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)

            connection = np.zeros((0, 6))
            for c in range(len(connection_candidate)):
                i, j, s, limb_len = connection_candidate[c][0:4]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j, limb_len]])
                    if len(connection) >= min(nA, nB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    return connection_all, special_k


def find_people(connection_all, special_k, all_peaks, limb_seq):
    len_rate = 16.0
    connection_tole = 0.7
    remove_recon = 0

    subset = -1 * np.ones((0, 20, 2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(limb_seq)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limb_seq[k])

            for i in range(len(connection_all[k])):
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    if subset[j][indexA][0].astype(int) == (partAs[i]).astype(int) or subset[j][indexB][0].astype(
                            int) == partBs[i].astype(int):
                        if found >= 2:
                            continue
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]

                    if subset[j][indexB][0].astype(int) == -1 and\
                            len_rate * subset[j][-1][1] > connection_all[k][i][-1]:
                        subset[j][indexB][0] = partBs[i]
                        subset[j][indexB][1] = connection_all[k][i][2]
                        subset[j][-1][0] += 1

                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                    elif subset[j][indexB][0].astype(int) != partBs[i].astype(int):
                        if subset[j][indexB][1] >= connection_all[k][i][2]:
                            pass

                        else:
                            if len_rate * subset[j][-1][1] <= connection_all[k][i][-1]:
                                continue
                            subset[j][-2][0] -= candidate[subset[j][indexB][0].astype(int), 2] + subset[j][indexB][1]

                            subset[j][indexB][0] = partBs[i]
                            subset[j][indexB][1] = connection_all[k][i][2]
                            subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                            subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                    elif subset[j][indexB][0].astype(int) == partBs[i].astype(int) and\
                            subset[j][indexB][1] <= connection_all[k][i][2]:
                        subset[j][-2][0] -= candidate[subset[j][indexB][0].astype(int), 2] + subset[j][indexB][1]

                        subset[j][indexB][0] = partBs[i]
                        subset[j][indexB][1] = connection_all[k][i][2]
                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                    else:
                        pass

                elif found == 2:
                    j1, j2 = subset_idx

                    membership1 = ((subset[j1][..., 0] >= 0).astype(int))[:-2]
                    membership2 = ((subset[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    if len(np.nonzero(membership == 2)[0]) == 0:

                        min_limb1 = np.min(subset[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(subset[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)

                        if connection_all[k][i][2] < connection_tole * min_tolerance or\
                                len_rate * subset[j1][-1][1] <= connection_all[k][i][-1]:
                            continue

                        subset[j1][:-2][...] += (subset[j2][:-2][...] + 1)

                        subset[j1][-2:][:, 0] += subset[j2][-2:][:, 0]

                        subset[j1][-2][0] += connection_all[k][i][2]
                        subset[j1][-1][1] = max(connection_all[k][i][-1], subset[j1][-1][1])
                        subset = np.delete(subset, j2, 0)

                    else:
                        if connection_all[k][i][0] in subset[j1, :-2, 0]:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][0])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][1])
                        else:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][1])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][0])

                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        if connection_all[k][i][2] < subset[j1][c1][1] and connection_all[k][i][2] < subset[j2][c2][1]:
                            continue

                        small_j = j1
                        remove_c = c1

                        if subset[j1][c1][1] > subset[j2][c2][1]:
                            small_j = j2
                            remove_c = c2

                        if remove_recon > 0:
                            subset[small_j][-2][0] -= candidate[subset[small_j][remove_c][0].astype(int), 2] + \
                                                      subset[small_j][remove_c][1]
                            subset[small_j][remove_c][0] = -1
                            subset[small_j][remove_c][1] = -1
                            subset[small_j][-1][0] -= 1

                elif not found and k < len(limb_seq):
                    row = -1 * np.ones((20, 2))
                    row[indexA][0] = partAs[i]
                    row[indexA][1] = connection_all[k][i][2]
                    row[indexB][0] = partBs[i]
                    row[indexB][1] = connection_all[k][i][2]
                    row[-1][0] = 2
                    row[-1][1] = connection_all[k][i][-1]
                    row[-2][0] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    row = row[np.newaxis, :, :]
                    subset = np.concatenate((subset, row), axis=0)
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1][0] < 2 or subset[i][-2][0] / subset[i][-1][0] < 0.45:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return subset, candidate

# ---------------------------------------------------------------------------------------------------------------------


class CocoHpe3MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CocoHpe3MetaInfo, self).__init__()
        self.label = "COCO"
        self.short_label = "coco"
        self.root_dir_name = "coco"
        self.dataset_class = CocoHpe3Dataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = 17
        self.input_image_size = (256, 256)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_capts = ["Val.CocoOksAp"]
        self.test_metric_names = ["CocoHpeOksApMetric"]
        self.test_metric_extra_kwargs = [
            {"name": "OksAp",
             "coco_annotations_file_path": None,
             "validation_ids": None,
             "use_file": False,
             "pose_postprocessing_fn": lambda x, y: recalc_pose(x, y)}]
        self.saver_acc_ind = 0
        self.do_transform = True
        self.val_transform = CocoHpe2ValTransform
        self.test_transform = CocoHpe2ValTransform
        self.ml_type = "hpe"
        self.test_net_extra_kwargs = None
        self.mean_rgb = (0.485, 0.456, 0.406)
        self.std_rgb = (0.229, 0.224, 0.225)
        self.load_ignore_extra = False

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        """
        Create python script parameters (for ImageNet-1K dataset metainfo).

        Parameters:
        ----------
        parser : ArgumentParser
            ArgumentParser instance.
        work_dir_path : str
            Path to working directory.
        """
        super(CocoHpe3MetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--input-size",
            type=int,
            nargs=2,
            default=self.input_image_size,
            help="size of the input for model")
        parser.add_argument(
            "--load-ignore-extra",
            action="store_true",
            help="ignore extra layers in the source PyTroch model")

    def update(self,
               args):
        """
        Update ImageNet-1K dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        super(CocoHpe3MetaInfo, self).update(args)
        self.input_image_size = args.input_size
        self.load_ignore_extra = args.load_ignore_extra

    def update_from_dataset(self,
                            dataset):
        """
        Update dataset metainfo after a dataset class instance creation.

        Parameters:
        ----------
        args : obj
            A dataset class instance.
        """
        self.test_metric_extra_kwargs[0]["coco_annotations_file_path"] = dataset.annotations_file_path
        # self.test_metric_extra_kwargs[0]["validation_ids"] = dataset.validation_ids
