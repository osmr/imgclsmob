"""
    COCO keypoint detection (2D single human pose estimation) dataset.
"""

import os
import copy
import cv2
import numpy as np
import torch
import torch.utils.data as data
from .dataset_metainfo import DatasetMetaInfo


class CocoHpe1Dataset(data.Dataset):
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
    CLASSES = ["person"]
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
        super(CocoHpe1Dataset, self).__init__()
        self._root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform
        self.num_class = len(self.CLASSES)

        if isinstance(splits, str):
            splits = [splits]
        self._splits = splits
        self._coco = []
        self._check_centers = check_centers
        self._skip_empty = skip_empty
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._items, self._labels = self._load_jsons()

        mode_name = "train" if mode == "train" else "val"
        annotations_dir_path = os.path.join(root, "annotations")
        annotations_file_path = os.path.join(annotations_dir_path, "person_keypoints_" + mode_name + "2017.json")
        self.annotations_file_path = annotations_file_path

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
        return [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

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
        # img = mx.image.imread(img_path, 1)
        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img, scale, center, score = self.transform(img, label)

        res_label = np.array([float(img_id)] + [float(score)] + list(center) + list(scale), np.float32)

        img = torch.from_numpy(img)
        res_label = torch.from_numpy(res_label)

        return img, res_label

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

# ---------------------------------------------------------------------------------------------------------------------


class CocoHpeValTransform1(object):
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
        # src_np = np.array(src)
        img = cv2.warpAffine(src, trans, (int(w), int(h)), flags=cv2.INTER_LINEAR)

        # img = mx.nd.image.to_tensor(mx.nd.array(img))
        # img = mx.nd.image.normalize(img, mean=self.mean, std=self.std)
        img = img.astype(np.float32)
        img = img / 255.0
        img = (img - np.array(self.mean, np.float32)) / np.array(self.std, np.float32)
        img = img.transpose((2, 0, 1))

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

# ---------------------------------------------------------------------------------------------------------------------


class CocoHpeValTransform2(object):
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
        # print(src.shape)
        bbox = label["bbox"]
        assert len(bbox) == 4
        score = label.get('score', 1)
        img, scale_box = detector_to_alpha_pose(
            src,
            class_ids=np.array([[0.]]),
            scores=np.array([[1.]]),
            bounding_boxs=np.array(np.array([bbox])),
            output_shape=self.image_size)

        if scale_box.shape[0] == 1:
            pt1 = np.array(scale_box[0, (0, 1)], dtype=np.float32)
            pt2 = np.array(scale_box[0, (2, 3)], dtype=np.float32)
        else:
            assert scale_box.shape[0] == 4
            pt1 = np.array(scale_box[(0, 1)], dtype=np.float32)
            pt2 = np.array(scale_box[(2, 3)], dtype=np.float32)

        return img[0].astype(np.float32), pt1, pt2, score


def detector_to_alpha_pose(img,
                           class_ids,
                           scores,
                           bounding_boxs,
                           output_shape=(256, 192),
                           thr=0.5):
    boxes, scores = alpha_pose_detection_processor(
        img=img,
        boxes=bounding_boxs,
        class_idxs=class_ids,
        scores=scores,
        thr=thr)
    pose_input, upscale_bbox = alpha_pose_image_cropper(
        source_img=img,
        boxes=boxes,
        output_shape=output_shape)
    return pose_input, upscale_bbox


def alpha_pose_detection_processor(img,
                                   boxes,
                                   class_idxs,
                                   scores,
                                   thr=0.5):
    if len(boxes.shape) == 3:
        boxes = boxes.squeeze(axis=0)
    if len(class_idxs.shape) == 3:
        class_idxs = class_idxs.squeeze(axis=0)
    if len(scores.shape) == 3:
        scores = scores.squeeze(axis=0)

    # cilp coordinates
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., img.shape[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., img.shape[0] - 1)

    # select boxes
    mask1 = (class_idxs == 0).astype(np.int32)
    mask2 = (scores > thr).astype(np.int32)
    picked_idxs = np.where((mask1 + mask2) > 1)[0]
    if picked_idxs.shape[0] == 0:
        return None, None
    else:
        return boxes[picked_idxs], scores[picked_idxs]


def alpha_pose_image_cropper(source_img,
                             boxes,
                             output_shape=(256, 192)):
    if boxes is None:
        return None, boxes

    # crop person poses
    img_width, img_height = source_img.shape[1], source_img.shape[0]

    tensors = np.zeros([boxes.shape[0], 3, output_shape[0], output_shape[1]])
    out_boxes = np.zeros([boxes.shape[0], 4])

    for i, box in enumerate(boxes):
        img = source_img.copy()
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        if box_width > 100:
            scale_rate = 0.2
        else:
            scale_rate = 0.3

        # crop image
        left = int(max(0, box[0] - box_width * scale_rate / 2))
        up = int(max(0, box[1] - box_height * scale_rate / 2))
        right = int(min(img_width - 1, max(left + 5, box[2] + box_width * scale_rate / 2)))
        bottom = int(min(img_height - 1, max(up + 5, box[3] + box_height * scale_rate / 2)))
        crop_width = right - left
        if crop_width < 1:
            continue
        crop_height = bottom - up
        if crop_height < 1:
            continue
        ul = np.array((left, up))
        br = np.array((right, bottom))
        img = cv_cropBox(img, ul, br, output_shape[0], output_shape[1])

        img = img.astype(np.float32)
        img = img / 255.0
        img = img.transpose((2, 0, 1))
        # img = mx.nd.image.to_tensor(np.array(img))
        # img = img.transpose((2, 0, 1))
        img[0] = img[0] - 0.406
        img[1] = img[1] - 0.457
        img[2] = img[2] - 0.480
        assert (img.shape[0] == 3)
        tensors[i] = img
        out_boxes[i] = (left, up, right, bottom)

    return tensors, out_boxes


def cv_cropBox(img, ul, br, resH, resW, pad_val=0):
    ul = ul
    br = (br - 1)
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.ndim == 2:
        img = img[:, np.newaxis]

    box_shape = [br[1] - ul[1], br[0] - ul[0]]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    img[:ul[1], :, :], img[:, :ul[0], :] = pad_val, pad_val
    img[br[1] + 1:, :, :], img[:, br[0] + 1:, :] = pad_val, pad_val

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array([ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array([br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(img, trans, (resW, resH), flags=cv2.INTER_LINEAR)

    return dst_img

# ---------------------------------------------------------------------------------------------------------------------


def recalc_pose1(keypoints,
                 bbs,
                 image_size):

    def transform_preds(coords, center, scale, output_size):

        def affine_transform(pt, t):
            new_pt = np.array([pt[0], pt[1], 1.]).T
            new_pt = np.dot(t, new_pt)
            return new_pt[:2]

        target_coords = np.zeros(coords.shape)
        trans = get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        return target_coords

    center = bbs[:, :2]
    scale = bbs[:, 2:4]

    heatmap_height = image_size[0] // 4
    heatmap_width = image_size[1] // 4
    output_size = [heatmap_width, heatmap_height]

    preds = np.zeros_like(keypoints)

    for i in range(keypoints.shape[0]):
        preds[i] = transform_preds(keypoints[i], center[i], scale[i], output_size)

    return preds


def recalc_pose1b(pred,
                  label,
                  image_size,
                  visible_conf_threshold=0.0):
    label_img_id = label[:, 0].astype(np.int32)
    label_score = label[:, 1]

    label_bbs = label[:, 2:6]
    pred_keypoints = pred[:, :, :2]
    pred_score = pred[:, :, 2]

    pred[:, :, :2] = recalc_pose1(pred_keypoints, label_bbs, image_size)
    pred_person_score = []

    batch = pred_keypoints.shape[0]
    num_joints = pred_keypoints.shape[1]
    for idx in range(batch):
        kpt_score = 0
        count = 0
        for i in range(num_joints):
            mval = float(pred_score[idx][i])
            if mval > visible_conf_threshold:
                kpt_score += mval
                count += 1

        if count > 0:
            kpt_score /= count

        kpt_score = kpt_score * float(label_score[idx])

        pred_person_score.append(kpt_score)

    return pred, pred_person_score, label_img_id


def recalc_pose2(keypoints,
                 bbs,
                 image_size):

    def transformBoxInvert(pt, ul, br, resH, resW):
        center = np.zeros(2)
        center[0] = (br[0] - 1 - ul[0]) / 2
        center[1] = (br[1] - 1 - ul[1]) / 2

        lenH = max(br[1] - ul[1], (br[0] - ul[0]) * resH / resW)
        lenW = lenH * resW / resH

        _pt = (pt * lenH) / resH

        if bool(((lenW - 1) / 2 - center[0]) > 0):
            _pt[0] = _pt[0] - ((lenW - 1) / 2 - center[0])
        if bool(((lenH - 1) / 2 - center[1]) > 0):
            _pt[1] = _pt[1] - ((lenH - 1) / 2 - center[1])

        new_point = np.zeros(2)
        new_point[0] = _pt[0] + ul[0]
        new_point[1] = _pt[1] + ul[1]
        return new_point

    pt2 = bbs[:, :2]
    pt1 = bbs[:, 2:4]

    heatmap_height = image_size[0] // 4
    heatmap_width = image_size[1] // 4

    preds = np.zeros_like(keypoints)

    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            preds[i, j] = transformBoxInvert(keypoints[i, j], pt1[i], pt2[i], heatmap_height, heatmap_width)

    return preds


def recalc_pose2b(pred,
                  label,
                  image_size,
                  visible_conf_threshold=0.0):
    label_img_id = label[:, 0].astype(np.int32)
    label_score = label[:, 1]

    label_bbs = label[:, 2:6]
    pred_keypoints = pred[:, :, :2]
    pred_score = pred[:, :, 2]

    pred[:, :, :2] = recalc_pose2(pred_keypoints, label_bbs, image_size)
    pred_person_score = []

    batch = pred_keypoints.shape[0]
    num_joints = pred_keypoints.shape[1]
    for idx in range(batch):
        kpt_score = 0
        count = 0
        for i in range(num_joints):
            mval = float(pred_score[idx][i])
            if mval > visible_conf_threshold:
                kpt_score += mval
                count += 1

        if count > 0:
            kpt_score /= count

        kpt_score = kpt_score * float(label_score[idx])

        pred_person_score.append(kpt_score)

    return pred, pred_person_score, label_img_id

# ---------------------------------------------------------------------------------------------------------------------


class CocoHpe1MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CocoHpe1MetaInfo, self).__init__()
        self.label = "COCO"
        self.short_label = "coco"
        self.root_dir_name = "coco"
        self.dataset_class = CocoHpe1Dataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = CocoHpe1Dataset.classes
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
             "coco_annotations_file_path": None,
             "use_file": False,
             "pose_postprocessing_fn": lambda x, y: recalc_pose1b(x, y, self.input_image_size)}]
        self.saver_acc_ind = 0
        self.do_transform = True
        self.val_transform = CocoHpeValTransform1
        self.test_transform = CocoHpeValTransform1
        self.ml_type = "hpe"
        self.net_extra_kwargs = {}
        self.mean_rgb = (0.485, 0.456, 0.406)
        self.std_rgb = (0.229, 0.224, 0.225)
        self.model_type = 1

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
        super(CocoHpe1MetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--input-size",
            type=int,
            nargs=2,
            default=self.input_image_size,
            help="size of the input for model")
        parser.add_argument(
            "--model-type",
            type=int,
            default=self.model_type,
            help="model type (1=SimplePose, 2=AlphaPose)")

    def update(self,
               args):
        """
        Update ImageNet-1K dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        super(CocoHpe1MetaInfo, self).update(args)
        self.input_image_size = args.input_size
        self.model_type = args.model_type
        if self.model_type == 1:
            self.test_metric_extra_kwargs[0]["pose_postprocessing_fn"] =\
                lambda x, y: recalc_pose1b(x, y, self.input_image_size)
            self.val_transform = CocoHpeValTransform1
            self.test_transform = CocoHpeValTransform1
        else:
            self.test_metric_extra_kwargs[0]["pose_postprocessing_fn"] =\
                lambda x, y: recalc_pose2b(x, y, self.input_image_size)
            self.val_transform = CocoHpeValTransform2
            self.test_transform = CocoHpeValTransform2

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
