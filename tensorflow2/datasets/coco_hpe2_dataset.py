"""
    COCO keypoint detection (2D multiple human pose estimation) dataset (for Lightweight OpenPose).
"""

import os
import json
import math
import threading
import cv2
from operator import itemgetter
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from .dataset_metainfo import DatasetMetaInfo


class CocoHpe2Dataset(object):
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
        super(CocoHpe2Dataset, self).__init__()
        self._root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform

        mode_name = "train" if mode == "train" else "val"
        annotations_dir_path = os.path.join(root, "annotations")
        annotations_file_path = os.path.join(annotations_dir_path, "person_keypoints_" + mode_name + "2017.json")
        with open(annotations_file_path, "r") as f:
            self.file_names = json.load(f)["images"]
        self.image_dir_path = os.path.join(root, mode_name + "2017")
        self.annotations_file_path = annotations_file_path

    def __str__(self):
        return self.__class__.__name__ + "(" + self._root + ")"

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]["file_name"]
        image_file_path = os.path.join(self.image_dir_path, file_name)
        image = cv2.imread(image_file_path, flags=cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

        img_mean = (128, 128, 128)
        img_scale = 1.0 / 256
        base_height = 368
        stride = 8
        pad_value = (0, 0, 0)

        height, width, _ = image.shape
        image = self.normalize(image, img_mean, img_scale)
        ratio = base_height / float(image.shape[0])
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(image.shape[1], base_height)]
        image, pad = self.pad_width(
            image,
            stride,
            pad_value,
            min_dims)
        image = image.astype(np.float32)
        # image = image.transpose((2, 0, 1))
        # image = torch.from_numpy(image)

        # if self.transform is not None:
        #     image = self.transform(image)

        image_id = int(os.path.splitext(os.path.basename(file_name))[0])

        label = np.array([image_id, 1.0] + pad + [height, width], np.float32)
        # label = torch.from_numpy(label)

        return image, label

    @staticmethod
    def normalize(img,
                  img_mean,
                  img_scale):
        img = np.array(img, dtype=np.float32)
        img = (img - img_mean) * img_scale
        return img

    @staticmethod
    def pad_width(img,
                  stride,
                  pad_value,
                  min_dims):
        h, w, _ = img.shape
        h = min(min_dims[0], h)
        min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
        min_dims[1] = max(min_dims[1], w)
        min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
        top = int(math.floor((min_dims[0] - h) / 2.0))
        left = int(math.floor((min_dims[1] - w) / 2.0))
        bottom = int(min_dims[0] - h - top)
        right = int(min_dims[1] - w - left)
        pad = [top, left, bottom, right]
        padded_img = cv2.copyMakeBorder(
            src=img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value)
        return padded_img, pad

# ---------------------------------------------------------------------------------------------------------------------


class CocoHpe2ValTransform(object):
    def __init__(self,
                 ds_metainfo):
        self.ds_metainfo = ds_metainfo

    def __call__(self, src, label):
        return src, label


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


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def recalc_pose(pred,
                label):
    label_img_id = label[:, 0].astype(np.int32)
    # label_score = label[:, 1]

    pred = pred.transpose((0, 3, 1, 2))

    pads = label[:, 2:6].astype(np.int32)
    heights = label[:, 6].astype(np.int32)
    widths = label[:, 7].astype(np.int32)

    keypoints = 19
    stride = 8

    heatmap2ds = pred[:, :keypoints]
    paf2ds = pred[:, keypoints:(3 * keypoints)]

    pred_pts_score = []
    pred_person_score = []
    label_img_id_ = []

    batch = pred.shape[0]
    for batch_i in range(batch):
        label_img_id_i = label_img_id[batch_i]
        pad = list(pads[batch_i])
        height = int(heights[batch_i])
        width = int(widths[batch_i])
        heatmap2d = heatmap2ds[batch_i]
        paf2d = paf2ds[batch_i]

        heatmaps = np.transpose(heatmap2d, (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)

        pafs = np.transpose(paf2d, (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx],
                all_keypoints_by_type,
                total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type,
            pafs)

        coco_keypoints, scores = convert_to_coco_format(
            pose_entries,
            all_keypoints)

        pred_pts_score.append(coco_keypoints)
        pred_person_score.append(scores)
        label_img_id_.append([label_img_id_i] * len(scores))

    return np.array(pred_pts_score).reshape((-1, 17, 3)), np.array(pred_person_score)[0], np.array(label_img_id_[0])

# ---------------------------------------------------------------------------------------------------------------------


class CocoHpe2MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CocoHpe2MetaInfo, self).__init__()
        self.label = "COCO"
        self.short_label = "coco"
        self.root_dir_name = "coco"
        self.dataset_class = CocoHpe2Dataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = 17
        self.input_image_size = (368, 368)
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
             "use_file": False,
             "pose_postprocessing_fn": lambda x, y: recalc_pose(x, y)}]
        self.saver_acc_ind = 0
        self.do_transform = True
        self.test_transform = cocohpe_val_transform
        self.test_transform2 = CocoHpe2ValTransform
        self.test_generator = cocohpe_test_generator
        self.ml_type = "hpe"
        self.net_extra_kwargs = {}
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
        super(CocoHpe2MetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
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
        super(CocoHpe2MetaInfo, self).update(args)
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

# ---------------------------------------------------------------------------------------------------------------------


class CocoHpeDirectoryIterator(DirectoryIterator):
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(368, 368),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 dataset=None):
        super(CocoHpeDirectoryIterator, self).set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation)

        self.dataset = dataset
        self.class_mode = class_mode
        self.dtype = dtype

        self.n = len(self.dataset)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = None
        batch_y = None
        for i, j in enumerate(index_array):
            x, y = self.dataset[j]
            if batch_x is None:
                batch_x = np.zeros((len(index_array),) + x.shape, dtype=self.dtype)
                batch_y = np.zeros((len(index_array),) + y.shape, dtype=np.float32)
            batch_x[i] = x
            batch_y[i] = y
        return batch_x, batch_y


class CocoHpeImageDataGenerator(ImageDataGenerator):

    def flow_from_directory(self,
                            directory,
                            target_size=(368, 368),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest',
                            dataset=None):
        return CocoHpeDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            dataset=dataset)


def cocohpe_val_transform(ds_metainfo,
                          data_format="channels_last"):
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = CocoHpeImageDataGenerator(
        preprocessing_function=(lambda img: ds_metainfo.val_transform2(ds_metainfo=ds_metainfo)(img)),
        data_format=data_format)
    return data_generator


def cocohpe_val_generator(data_generator,
                          ds_metainfo,
                          batch_size):
    """
    Create image generator for validation subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    split = "val"
    root = ds_metainfo.root_dir_path
    root = os.path.join(root, split)
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation="bilinear",
        dataset=ds_metainfo.dataset_class(
            root=ds_metainfo.root_dir_path,
            mode="val",
            transform=ds_metainfo.val_transform2(
                ds_metainfo=ds_metainfo)))
    return generator


def cocohpe_test_generator(data_generator,
                           ds_metainfo,
                           batch_size):
    """
    Create image generator for testing subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    split = "val"
    root = ds_metainfo.root_dir_path
    root = os.path.join(root, split)
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation="bilinear",
        dataset=ds_metainfo.dataset_class(
            root=ds_metainfo.root_dir_path,
            mode="test",
            transform=ds_metainfo.test_transform2(
                ds_metainfo=ds_metainfo)))
    return generator
