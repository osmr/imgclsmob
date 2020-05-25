"""
MS COCO object detection dataset.
"""

__all__ = ['CocoDetMetaInfo']

import os
import cv2
import logging
import mxnet as mx
import numpy as np
from PIL import Image
from mxnet.gluon.data import dataset
from .dataset_metainfo import DatasetMetaInfo


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
        self._transform = transform
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

        mode_name = "train" if mode == "train" else "val"
        annotations_dir_path = os.path.join(root, "annotations")
        annotations_file_path = os.path.join(annotations_dir_path, "instances_" + mode_name + "2017.json")
        self.annotations_file_path = annotations_file_path

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
        dirname, filename = entry["coco_url"].split("/")[-2:]
        abs_path = os.path.join(self._root, dirname, filename)
        return abs_path

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]
        img = mx.image.imread(img_path, 1)
        label = np.array(label).copy()
        if self._transform is not None:
            img, label = self._transform(img, label)
        return img, label

    def _load_jsons(self):
        """
        Load all image paths and labels from JSON annotation files into buffer.
        """
        items = []
        labels = []
        im_aspect_ratios = []

        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self._root, self.annotation_dir, split) + ".json"
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
                abs_path = self._parse_image_path(entry)
                if not os.path.exists(abs_path):
                    raise IOError("Image: {} not exists.".format(abs_path))
                label = self._check_load_bbox(_coco, entry)
                if not label:
                    continue
                im_aspect_ratios.append(float(entry["width"]) / entry["height"])
                items.append(abs_path)
                labels.append(label)
        return items, labels, im_aspect_ratios

    def _check_load_bbox(self, coco, entry):
        """
        Check and load ground-truth labels.
        """
        entry_id = entry['id']
        # fix pycocotools _isArrayLike which don't work for str in python3
        entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
        ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry["width"]
        height = entry["height"]
        for obj in objs:
            if obj["area"] < self._min_object_area:
                continue
            if obj.get("ignore", 0) == 1:
                continue
            if not self._use_crowd and obj.get("iscrowd", 0):
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = self.bbox_clip_xyxy(self.bbox_xywh_to_xyxy(obj["bbox"]), width, height)
            # require non-zero box area
            if obj["area"] > 0 and xmax > xmin and ymax > ymin:
                contiguous_cid = self.json_id_to_contiguous[obj["category_id"]]
                valid_objs.append([xmin, ymin, xmax, ymax, contiguous_cid])
        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append([-1, -1, -1, -1, -1])
        return valid_objs

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
        self.image_size = self.ds_metainfo.input_image_size
        self._height = self.image_size[0]
        self._width = self.image_size[1]
        self._mean = np.array(ds_metainfo.mean_rgb, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(ds_metainfo.std_rgb, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, src, label):
        # resize
        img, bbox = src.asnumpy(), label
        input_h, input_w = self._height, self._width
        h, w, _ = src.shape
        s = max(h, w) * 1.0
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        trans_input = self.get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        output_w = input_w
        output_h = input_h
        trans_output = self.get_affine_transform(c, s, 0, [output_w, output_h])
        for i in range(bbox.shape[0]):
            bbox[i, :2] = self.affine_transform(bbox[i, :2], trans_output)
            bbox[i, 2:4] = self.affine_transform(bbox[i, 2:4], trans_output)
        bbox[:, :2] = np.clip(bbox[:, :2], 0, output_w - 1)
        bbox[:, 2:4] = np.clip(bbox[:, 2:4], 0, output_h - 1)
        img = inp

        # to tensor
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = mx.nd.array(img)
        return img, bbox.astype(img.dtype)

    @staticmethod
    def get_affine_transform(center,
                             scale,
                             rot,
                             output_size,
                             shift=np.array([0, 0], dtype=np.float32),
                             inv=0):
        """
        Get affine transform matrix given center, scale and rotation.

        Parameters
        ----------
        center : tuple of float
            Center point.
        scale : float
            Scaling factor.
        rot : float
            Rotation degree.
        output_size : tuple of int
            (width, height) of the output size.
        shift : float
            Shift factor.
        inv : bool
            Whether inverse the computation.

        Returns
        -------
        numpy.ndarray
            Affine matrix.
        """
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = CocoDetValTransform.get_rot_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = CocoDetValTransform.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = CocoDetValTransform.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    @staticmethod
    def get_rot_dir(src_point, rot_rad):
        """
        Get rotation direction.

        Parameters
        ----------
        src_point : tuple of float
            Original point.
        rot_rad : float
            Rotation radian.

        Returns
        -------
        tuple of float
            Rotation.
        """
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    @staticmethod
    def get_3rd_point(a, b):
        """
        Get the 3rd point position given first two points.

        Parameters
        ----------
        a : tuple of float
            First point.
        b : tuple of float
            Second point.

        Returns
        -------
        tuple of float
            Third point.
        """
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    @staticmethod
    def affine_transform(pt, t):
        """
        Apply affine transform to a bounding box given transform matrix t.

        Parameters
        ----------
        pt : numpy.ndarray
            Bounding box with shape (1, 2).
        t : numpy.ndarray
            Transformation matrix with shape (2, 3).

        Returns
        -------
        numpy.ndarray
            New bounding box with shape (1, 2).
        """
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


class Tuple(object):
    """
    Wrap multiple batchify functions to form a function apply each input function on each
    input fields respectively.
    """
    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            self._fn = fn
        else:
            self._fn = (fn,) + args

    def __call__(self, data):
        """
        Batchify the input data.

        Parameters
        ----------
        data : list
            The samples to batchfy. Each sample should contain N attributes.
        Returns
        -------
        tuple
            A tuple of length N. Contains the batchified result of each attribute in the input.
        """
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)


class Stack(object):
    """
    Stack the input data samples to construct the batch.
    """
    def __call__(self, data):
        """
        Batchify the input data.

        Parameters
        ----------
        data : list
            The input data samples

        Returns
        -------
        NDArray
            Result.
        """
        return self._stack_arrs(data, True)

    @staticmethod
    def _stack_arrs(arrs, use_shared_mem=False):
        """
        Internal imple for stacking arrays.
        """
        if isinstance(arrs[0], mx.nd.NDArray):
            if use_shared_mem:
                out = mx.nd.empty((len(arrs),) + arrs[0].shape, dtype=arrs[0].dtype,
                                  ctx=mx.Context("cpu_shared", 0))
                return mx.nd.stack(*arrs, out=out)
            else:
                return mx.nd.stack(*arrs)
        else:
            out = np.asarray(arrs)
            if use_shared_mem:
                return mx.nd.array(out, ctx=mx.Context("cpu_shared", 0))
            else:
                return mx.nd.array(out)


class Pad(object):
    """
    Pad the input ndarrays along the specific padding axis and stack them to get the output.
    """
    def __init__(self, axis=0, pad_val=0, num_shards=1, ret_length=False):
        self._axis = axis
        self._pad_val = pad_val
        self._num_shards = num_shards
        self._ret_length = ret_length

    def __call__(self, data):
        """
        Batchify the input data.

        Parameters
        ----------
        data : list
            A list of N samples. Each sample can be 1) ndarray or
             2) a list/tuple of ndarrays
        Returns
        -------
        NDArray
            Data in the minibatch. Shape is (N, ...)
        NDArray, optional
            The sequences' original lengths at the padded axis. Shape is (N,). This will only be
            returned in `ret_length` is True.
        """
        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = self._pad_arrs_to_max_length(
                data, self._axis, self._pad_val, self._num_shards, True)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr

        else:
            raise NotImplementedError

    @staticmethod
    def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, num_shards=1, use_shared_mem=False):
        """
        Inner Implementation of the Pad batchify.
        """
        if not isinstance(arrs[0], (mx.nd.NDArray, np.ndarray)):
            arrs = [np.asarray(ele) for ele in arrs]
        if isinstance(pad_axis, tuple):
            original_length = []
            for axis in pad_axis:
                original_length.append(np.array([ele.shape[axis] for ele in arrs]))
            original_length = np.stack(original_length).T
        else:
            original_length = np.array([ele.shape[pad_axis] for ele in arrs])
            pad_axis = [pad_axis]
        if len(original_length) % num_shards != 0:
            logging.warning(
                'Batch size cannot be evenly split. Trying to shard %d items into %d shards',
                len(original_length), num_shards)
        original_length = np.array_split(original_length, num_shards)
        max_lengths = [np.max(ll, axis=0, keepdims=len(pad_axis) == 1) for ll in original_length]
        # add batch dimension
        ret_shape = [[ll.shape[0], ] + list(arrs[0].shape) for ll in original_length]
        for i, shape in enumerate(ret_shape):
            for j, axis in enumerate(pad_axis):
                shape[1 + axis] = max_lengths[i][j]
        if use_shared_mem:
            ret = [mx.nd.full(shape=tuple(shape), val=pad_val, ctx=mx.Context('cpu_shared', 0),
                              dtype=arrs[0].dtype) for shape in ret_shape]
            original_length = [mx.nd.array(ll, ctx=mx.Context('cpu_shared', 0),
                                           dtype=np.int32) for ll in original_length]
        else:
            ret = [mx.nd.full(shape=tuple(shape), val=pad_val, dtype=arrs[0].dtype) for shape in
                   ret_shape]
            original_length = [mx.nd.array(ll, dtype=np.int32) for ll in original_length]
        for i, arr in enumerate(arrs):
            if ret[i // ret[0].shape[0]].shape[1:] == arr.shape:
                ret[i // ret[0].shape[0]][i % ret[0].shape[0]] = arr
            else:
                slices = [slice(0, ll) for ll in arr.shape]
                ret[i // ret[0].shape[0]][i % ret[0].shape[0]][tuple(slices)] = arr
        if len(ret) == len(original_length) == 1:
            return ret[0], original_length[0]
        return ret, original_length


def get_post_transform(orig_w, orig_h, out_w, out_h):
    """Get the post prediction affine transforms. This will be used to adjust the prediction results
    according to original coco image resolutions.

    Parameters
    ----------
    orig_w : int
        Original width of the image.
    orig_h : int
        Original height of the image.
    out_w : int
        Width of the output image after prediction.
    out_h : int
        Height of the output image after prediction.

    Returns
    -------
    numpy.ndarray
        Affine transform matrix 3x2.

    """
    s = max(orig_w, orig_h) * 1.0
    c = np.array([orig_w / 2., orig_h / 2.], dtype=np.float32)
    trans_output = CocoDetValTransform.get_affine_transform(c, s, 0, [out_w, out_h], inv=True)
    return trans_output


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
        self.input_image_size = (512, 512)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_capts = ["Val.mAP"]
        self.test_metric_names = ["CocoDetMApMetric"]
        self.test_metric_extra_kwargs = [
            {"name": "mAP",
             "img_height": 512,
             "coco_annotations_file_path": None,
             "contiguous_id_to_json": None,
             "data_shape": None,
             "post_affine": get_post_transform}]
        self.test_dataset_extra_kwargs = {
            "skip_empty": False}
        self.saver_acc_ind = 0
        self.do_transform = True
        self.do_transform_first = False
        self.last_batch = "keep"
        self.batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        self.val_transform = CocoDetValTransform
        self.test_transform = CocoDetValTransform
        self.ml_type = "det"
        self.allow_hybridize = False
        self.test_net_extra_kwargs = None
        self.mean_rgb = (0.485, 0.456, 0.406)
        self.std_rgb = (0.229, 0.224, 0.225)

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
        self.test_metric_extra_kwargs[0]["img_height"] = self.input_image_size[0]
        self.test_metric_extra_kwargs[0]["data_shape"] = self.input_image_size

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
        self.test_metric_extra_kwargs[0]["contiguous_id_to_json"] = dataset.contiguous_id_to_json
