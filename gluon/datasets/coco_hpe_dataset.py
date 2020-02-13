"""
    COCO keypoint detection (2D single human pose estimation) dataset.
"""

import os
import copy
import cv2
import numpy as np
import mxnet as mx
from mxnet.gluon.data import dataset
from .dataset_metainfo import DatasetMetaInfo


class CocoHpeDataset(dataset.Dataset):
    """
    COCO keypoint detection (2D single human pose estimation) dataset.

    Parameters
    ----------
    root : string
        Path to `annotations`, `train2017`, and `val2017` folders.
    mode : string, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    splits : list of str, default ['person_keypoints_val2017']
        Json annotations name.
        Candidates can be: person_keypoints_val2017, person_keypoints_train2017.
    check_centers : bool, default is False
        If true, will force check centers of bbox and keypoints, respectively.
        If centers are far away from each other, remove this label.
    skip_empty : bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.
    """
    CLASSES = ['person']
    KEYPOINTS = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
        [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 splits=("person_keypoints_val2017",),
                 check_centers=False,
                 skip_empty=True):
        super(CocoHpeDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform
        self.num_class = len(self.CLASSES)

        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        self._coco = []
        self._check_centers = check_centers
        self._skip_empty = skip_empty
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._items, self._labels = self._load_jsons()

    def __str__(self):
        detail = ",".join([str(s) for s in self._splits])
        return self.__class__.__name__ + "(" + detail + ")"

    @property
    def classes(self):
        """
        Category names.
        """
        return type(self).CLASSES

    @property
    def num_joints(self):
        """
        Dataset defined: number of joints provided.
        """
        return 17

    @property
    def joint_pairs(self):
        """
        Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally.
        """
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    @property
    def coco(self):
        """
        Return pycocotools object for evaluation purposes.
        """
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        if len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files".format(len(self._coco)))
        return self._coco[0]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        label = copy.deepcopy(self._labels[idx])
        img = mx.image.imread(img_path, 1)

        if self.transform is not None:
            img, scale, center, score = self.transform(img, label)
        return img, scale, center, score, img_id

    def _load_jsons(self):
        """
        Load all image paths and labels from JSON annotation files into buffer.
        """
        items = []
        labels = []

        from pycocotools.coco import COCO

        for split in self._splits:
            anno = os.path.join(self._root, "annotations", split) + ".json"
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c["name"] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous
            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                dirname, filename = entry["coco_url"].split("/")[-2:]
                abs_path = os.path.join(self._root, dirname, filename)
                if not os.path.exists(abs_path):
                    raise IOError("Image: {} not exists.".format(abs_path))
                label = self._check_load_keypoints(_coco, entry)
                if not label:
                    continue

                # num of items are relative to person, not image
                for obj in label:
                    items.append(abs_path)
                    labels.append(obj)
        return items, labels

    def _check_load_keypoints(self, coco, entry):
        """
        Check and load ground-truth keypoints.
        """
        ann_ids = coco.getAnnIds(imgIds=entry["id"], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry["width"]
        height = entry["height"]

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj["category_id"]]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj["keypoints"]) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = self.bbox_clip_xyxy(self.bbox_xywh_to_xyxy(obj["bbox"]), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue

            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj["keypoints"][i * 3 + 0]
                joints_3d[i, 1, 0] = obj["keypoints"][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj["keypoints"][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                "bbox": (xmin, ymin, xmax, ymax),
                "joints_3d": joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    "bbox": np.array([-1, -1, 0, 0]),
                    "joints_3d": np.zeros((self.num_joints, 3, 2), dtype=np.float32)
                })
        return valid_objs

    @staticmethod
    def _get_box_center_area(bbox):
        """
        Get bbox center.
        """
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    @staticmethod
    def _get_keypoints_center_count(keypoints):
        """
        Get geometric center of all keypoints.
        """
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num

    @staticmethod
    def bbox_clip_xyxy(xyxy, width, height):
        """
        Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.

        All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

        Parameters
        ----------
        xyxy : list, tuple or numpy.ndarray
            The bbox in format (xmin, ymin, xmax, ymax).
            If numpy.ndarray is provided, we expect multiple bounding boxes with
            shape `(N, 4)`.
        width : int or float
            Boundary width.
        height : int or float
            Boundary height.

        Returns
        -------
        tuple or np.array
            Description of returned object.
        """
        if isinstance(xyxy, (tuple, list)):
            if not len(xyxy) == 4:
                raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
            x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
            y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
            x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
            y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
            return x1, y1, x2, y2
        elif isinstance(xyxy, np.ndarray):
            if not xyxy.size % 4 == 0:
                raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
            x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
            y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
            x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
            y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
            return np.hstack((x1, y1, x2, y2))
        else:
            raise TypeError("Expect input xywh a list, tuple or numpy.ndarray, given {}".format(type(xyxy)))

    @staticmethod
    def bbox_xywh_to_xyxy(xywh):
        """
        Convert bounding boxes from format (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)

        Parameters
        ----------
        xywh : list, tuple or numpy.ndarray
            The bbox in format (x, y, w, h).
            If numpy.ndarray is provided, we expect multiple bounding boxes with
            shape `(N, 4)`.

        Returns
        -------
        tuple or np.ndarray
            The converted bboxes in format (xmin, ymin, xmax, ymax).
            If input is numpy.ndarray, return is numpy.ndarray correspondingly.

        """
        if isinstance(xywh, (tuple, list)):
            if not len(xywh) == 4:
                raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xywh)))
            w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
            return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
        elif isinstance(xywh, np.ndarray):
            if not xywh.size % 4 == 0:
                raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
            xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
            return xyxy
        else:
            raise TypeError("Expect input xywh a list, tuple or numpy.ndarray, given {}".format(type(xywh)))


class CocoHpeValTransform(object):
    def __init__(self,
                 ds_metainfo):
        self.ds_metainfo = ds_metainfo
        self.image_size = self.ds_metainfo.input_image_size
        height = self.image_size[0]
        width = self.image_size[1]
        self.aspect_ratio = float(width / height)
        self.mean = ds_metainfo.mean_rgb
        self.std = ds_metainfo.std_rgb

    def __call__(self, src, label):
        bbox = label["bbox"]
        assert len(bbox) == 4
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin, self.aspect_ratio)
        score = label.get("score", 1)

        h, w = self.image_size
        trans = get_affine_transform(center, scale, 0, [w, h])
        img = cv2.warpAffine(src.asnumpy(), trans, (int(w), int(h)), flags=cv2.INTER_LINEAR)

        img = mx.nd.image.to_tensor(mx.nd.array(img))
        img = mx.nd.image.normalize(img, mean=self.mean, std=self.std)

        return img, scale, center, score


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    pixel_std = 1
    center = np.zeros((2,), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(
        img,
        trans,
        (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR)
    return dst_img


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class CocoHpeMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CocoHpeMetaInfo, self).__init__()
        self.label = "COCO"
        self.short_label = "coco"
        self.root_dir_name = "coco"
        self.dataset_class = CocoHpeDataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = CocoHpeDataset.classes
        self.input_image_size = (256, 192)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_capts = ["Val.CocoOksAp"]
        self.test_metric_names = ["CocoHpeOksApMetric"]
        self.test_metric_extra_kwargs = [
            {"name": "OksAp",
             "coco": None,
             "in_vis_thresh": 0.0}]
        self.saver_acc_ind = 0
        self.do_transform = True
        self.val_transform = CocoHpeValTransform
        self.test_transform = CocoHpeValTransform
        self.ml_type = "hpe"
        self.mean_rgb = (0.485, 0.456, 0.406)
        self.std_rgb = (0.229, 0.224, 0.225)

    def update_from_dataset(self,
                            dataset):
        """
        Update dataset metainfo after a dataset class instance creation.

        Parameters:
        ----------
        args : obj
            A dataset class instance.
        """
        self.test_metric_extra_kwargs[0]["coco"] = dataset.coco
