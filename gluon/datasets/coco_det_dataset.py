"""
MS COCO object detection dataset.
"""
import os
import mxnet as mx
import numpy as np
from PIL import Image
from mxnet.gluon.data import dataset
from .dataset_metainfo import DatasetMetaInfo

__all__ = ['CocoDetMetaInfo']


class CocoDetDataset(dataset.Dataset):
    """
    MS COCO detection dataset.

    Parameters
    ----------
    root : str
        Path to folder storing the dataset.
    mode : string, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    splits : list of str, default ['instances_val2017']
        Json annotations name.
        Candidates can be: instances_val2017, instances_train2017.
    min_object_area : float
        Minimum accepted ground-truth area, if an object's area is smaller than this value,
        it will be ignored.
    skip_empty : bool, default is True
        Whether skip images with no valid object. This should be `True` in training, otherwise
        it will cause undefined behavior.
    use_crowd : bool, default is True
        Whether use boxes labeled as crowd instance.
    """
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 splits=('instances_val2017',),
                 min_object_area=0,
                 skip_empty=True,
                 use_crowd=True):
        super(CocoDetDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform
        self.num_class = len(self.CLASSES)

        self._min_object_area = min_object_area
        self._skip_empty = skip_empty
        self._use_crowd = use_crowd
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._coco = []
        self._items, self._labels, self._im_aspect_ratios = self._load_jsons()

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def coco(self):
        """
        Return pycocotools object for evaluation purposes.
        """
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        if len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files. \
                Please use single JSON dataset and evaluate one by one".format(len(self._coco)))
        return self._coco[0]

    @property
    def classes(self):
        """
        Category names.
        """
        return type(self).CLASSES

    @property
    def annotation_dir(self):
        """
        The subdir for annotations. Default is 'annotations'(coco default)
        For example, a coco format json file will be searched as
        'root/annotation_dir/xxx.json'
        You can override if custom dataset don't follow the same pattern
        """
        return 'annotations'

    def get_im_aspect_ratio(self):
        """Return the aspect ratio of each image in the order of the raw data."""
        if self._im_aspect_ratios is not None:
            return self._im_aspect_ratios
        self._im_aspect_ratios = [None] * len(self._items)
        for i, img_path in enumerate(self._items):
            with Image.open(img_path) as im:
                w, h = im.size
                self._im_aspect_ratios[i] = 1.0 * w / h

        return self._im_aspect_ratios

    def _parse_image_path(self, entry):
        """How to parse image dir and path from entry.

        Parameters
        ----------
        entry : dict
            COCO entry, e.g. including width, height, image path, etc..

        Returns
        -------
        abs_path : str
            Absolute path for corresponding image.

        """
        dirname, filename = entry['coco_url'].split('/')[-2:]
        abs_path = os.path.join(self._root, dirname, filename)
        return abs_path

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]
        img = mx.image.imread(img_path, 1)
        if self.transform is not None:
            return self.transform(img, label)
        return img, np.array(label).copy()

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


class CocoDetValTransform(object):
    def __init__(self,
                 ds_metainfo):
        self.ds_metainfo = ds_metainfo

    def __call__(self, src, label):
        return src, label


class CocoDetMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CocoDetMetaInfo, self).__init__()
        self.label = "COCO"
        self.short_label = "coco"
        self.root_dir_name = "coco"
        self.dataset_class = CocoDetDataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = CocoDetDataset.classes
        self.input_image_size = (256, 192)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_capts = ["Val.mAP"]
        self.test_metric_names = ["CocoDetMApMetric"]
        self.test_metric_extra_kwargs = [
            {"name": "mAP",
             "coco_annotations_file_path": None,
             "use_file": False}]
        self.saver_acc_ind = 0
        self.do_transform = True
        self.val_transform = CocoDetValTransform
        self.test_transform = CocoDetValTransform
        self.ml_type = "hpe"
        self.allow_hybridize = False
        self.net_extra_kwargs = {"fixed_size": False}
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
        super(CocoDetMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
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
        super(CocoDetMetaInfo, self).update(args)
        self.input_image_size = args.input_size
        self.model_type = args.model_type

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
