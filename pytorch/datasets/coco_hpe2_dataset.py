"""
    COCO keypoint detection (2D multiple human pose estimation) dataset (for Lightweight OpenPose).
"""

import os
import json
import math
import cv2
import numpy as np
import torch
import torch.utils.data as data
from .dataset_metainfo import DatasetMetaInfo


class CocoHpe2Dataset(data.Dataset):
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
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        # if self.transform is not None:
        #     image = self.transform(image)

        image_id = int(os.path.splitext(os.path.basename(file_name))[0])

        label = np.array([image_id] + pad + [height, width], np.float32)
        label = torch.from_numpy(label)

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
        self.input_image_size = (256, 192)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_capts = ["Val.CocoOksAp"]
        self.test_metric_names = ["CocoHpe2OksApMetric"]
        self.test_metric_extra_kwargs = [
            {"name": "OksAp",
             "annotations_file_path": None}]
        self.saver_acc_ind = 0
        self.do_transform = True
        self.val_transform = CocoHpe2ValTransform
        self.test_transform = CocoHpe2ValTransform
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
        self.test_metric_extra_kwargs[0]["annotations_file_path"] = dataset.annotations_file_path
