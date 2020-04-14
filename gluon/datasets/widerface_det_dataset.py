"""
WIDER FACE detection dataset.
"""
import os
import cv2
import mxnet as mx
import numpy as np
from mxnet.gluon.data import dataset
from .dataset_metainfo import DatasetMetaInfo

__all__ = ['WiderfaceDetMetaInfo']


class WiderfaceDetDataset(dataset.Dataset):
    """
    WIDER FACE detection dataset.

    Parameters
    ----------
    root : str
        Path to folder storing the dataset.
    mode : string, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode="train",
                 transform=None):
        super(WiderfaceDetDataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.mode = mode
        self._transform = transform

        self.synsets = []
        self.items = []

        image_dir_path = "{}/WIDER_{}/images".format(self.root, self.mode)

        for folder in sorted(os.listdir(image_dir_path)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in (".jpg",):
                    continue
                self.items.append((filename, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx][0]
        # image = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        image = mx.image.imread(img_path, flag=1).asnumpy()

        image_size = image.shape[:2]

        shorter_side = min(image.shape[:2])
        resize_scale = 1.0
        if shorter_side < 128:
            resize_scale = 128.0 / shorter_side
        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)

        image = image.transpose(2, 0, 1).astype(np.float32)
        image = mx.nd.array(image)

        label = "{}/{}/{}/{}/{}".format(self.synsets[self.items[idx][1]], (img_path.split("/")[1]).split(".")[0],
                                        resize_scale, image_size[0], image_size[1])
        label = np.array(label).copy()

        if self._transform is not None:
            image, label = self._transform(image, label)
        return image, label

# ---------------------------------------------------------------------------------------------------------------------


class WiderfaceDetValTransform(object):
    def __init__(self,
                 ds_metainfo):
        self.ds_metainfo = ds_metainfo

    def __call__(self, image, label):
        return image, label

# ---------------------------------------------------------------------------------------------------------------------


class WiderfaceDetMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(WiderfaceDetMetaInfo, self).__init__()
        self.label = "WiderFace"
        self.short_label = "widerface"
        self.root_dir_name = "WIDER_FACE"
        self.dataset_class = WiderfaceDetDataset
        self.num_training_samples = None
        self.in_channels = 3
        self.input_image_size = (480, 640)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_capts = ["WF"]
        self.test_metric_names = ["WiderfaceDetMetric"]
        self.test_metric_extra_kwargs = [
            {"name": "WF"}]
        self.saver_acc_ind = 0
        self.do_transform = True
        self.do_transform_first = False
        self.last_batch = "keep"
        self.val_transform = WiderfaceDetValTransform
        self.test_transform = WiderfaceDetValTransform
        self.ml_type = "det"
        self.allow_hybridize = False
        self.test_net_extra_kwargs = None
        self.model_type = 1
        self.receptive_field_center_starts = None
        self.receptive_field_strides = None
        self.bbox_factors = None

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
        super(WiderfaceDetMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--model-type",
            type=int,
            default=self.model_type,
            help="model type (1=320, 2=560)")

    def update(self,
               args):
        """
        Update ImageNet-1K dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        super(WiderfaceDetMetaInfo, self).update(args)
        self.model_type = args.model_type
        if self.model_type == 1:
            self.receptive_field_center_starts = [3, 7, 15, 31, 63]
            self.receptive_field_strides = [4, 8, 16, 32, 64]
            self.bbox_factors = [10.0, 20.0, 40.0, 80.0, 160.0]
        else:
            self.receptive_field_center_starts = [3, 3, 7, 7, 15, 31, 31, 31]
            self.receptive_field_strides = [4, 4, 8, 8, 16, 32, 32, 32]
            self.bbox_factors = [7.5, 10.0, 20.0, 35.0, 55.0, 125.0, 200.0, 280.0]
