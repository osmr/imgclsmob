"""
    COCO keypoint detection (2D multiple human pose estimation) dataset (for IBPPose).
"""

import os
import json
# import math
import cv2
# from operator import itemgetter
import numpy as np
import torch
import torch.utils.data as data
from .dataset_metainfo import DatasetMetaInfo


class CocoHpe3Dataset(data.Dataset):
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

        max_downsample = 64
        pad_value = 128
        image, pad = self.pad_right_down_corner(image, max_downsample, pad_value)
        image = np.float32(image / 255)
        image = image[None, ...]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        image_id = int(os.path.splitext(os.path.basename(file_name))[0])

        label = np.array([image_id, 1.0] + pad, np.float32)
        label = torch.from_numpy(label)

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
    # label_img_id = label[:, 0].astype(np.int32)
    # label_score = label[:, 1]

    # pad = label[:, 2:6].astype(np.int32)
    #
    # paf_layers = 30
    # num_layers = 50
    # stride = 4
    #
    # output_blob = pred[0].transpose((1, 2, 0))
    # output_blob0 = output_blob[:, :, :paf_layers]
    # output_blob1 = output_blob[:, :, paf_layers:num_layers]
    #
    # output_blob0_avg = output_blob0
    # output_blob1_avg = output_blob1
    #
    # heatmap = cv2.resize(output_blob1_avg, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    #
    # heatmap = heatmap[pad[0]:imageToTest_padded.shape[0] - pad[2], pad[1]:imageToTest_padded.shape[1] - pad[3], :]
    # heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    #
    # # output_blob0 is PAFs
    # paf = cv2.resize(output_blob0_avg, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    #
    # paf = paf[pad[0]:imageToTest_padded.shape[0] - pad[2], pad[1]:imageToTest_padded.shape[1] - pad[3], :]
    # paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    #
    # heatmap_avg = heatmap_avg + heatmap / (len(multiplier) * len(rotate_angle))
    # paf_avg = paf_avg + paf / (len(multiplier) * len(rotate_angle))

    pred_pts_score = []
    pred_person_score = []
    label_img_id_ = []

    return np.array(pred_pts_score).reshape((-1, 17, 3)), np.array(pred_person_score)[0], np.array(label_img_id_[0])

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
