"""
    ImageNet-1K classification dataset.
"""

import os
import math
import numpy as np
from chainer.dataset import DatasetMixin
from chainercv.transforms import scale
from chainercv.transforms import center_crop
from chainercv.datasets import DirectoryParsingLabelDataset
from .dataset_metainfo import DatasetMetaInfo


class ImageNet1K(DatasetMixin):
    """
    ImageNet-1K classification dataset.

    Parameters
    ----------
    root : str, default '~/.chainer/datasets/imagenet'
        Path to the folder stored the dataset.
    mode: str, default 'train'
        'train', 'val', or 'test'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root=os.path.join("~", ".chainer", "datasets", "imagenet"),
                 mode="train",
                 transform=None):
        split = "train" if mode == "train" else "val"
        root = os.path.join(root, split)
        self.transform = transform
        self.base = DirectoryParsingLabelDataset(root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = self.transform(image)
        return image, label


class ImageNet1KMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(ImageNet1KMetaInfo, self).__init__()
        self.label = "ImageNet1K"
        self.short_label = "imagenet"
        self.root_dir_name = "imagenet"
        self.dataset_class = ImageNet1K
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = 1000
        self.input_image_size = (224, 224)
        self.resize_inv_factor = 0.875
        self.train_metric_capts = ["Train.Top1"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err-top1"}]
        self.val_metric_capts = ["Val.Top1", "Val.Top5"]
        self.val_metric_names = ["Top1Error", "TopKError"]
        self.val_metric_extra_kwargs = [{"name": "err-top1"}, {"name": "err-top5", "top_k": 5}]
        self.saver_acc_ind = 1
        self.train_transform = None
        self.val_transform = ImageNetValTransform
        self.test_transform = ImageNetValTransform
        self.ml_type = "imgcls"
        self.use_cv_resize = False

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        super(ImageNet1KMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--input-size",
            type=int,
            default=self.input_image_size[0],
            help="size of the input for model")
        parser.add_argument(
            "--resize-inv-factor",
            type=float,
            default=self.resize_inv_factor,
            help="inverted ratio for input image crop")

    def update(self,
               args):
        super(ImageNet1KMetaInfo, self).update(args)
        self.input_image_size = (args.input_size, args.input_size)


class ImageNetValTransform(object):
    """
    ImageNet-1K validation transform.
    """
    def __init__(self,
                 ds_metainfo,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.input_image_size = ds_metainfo.input_image_size
        self.resize_value = calc_val_resize_value(
            input_image_size=ds_metainfo.input_image_size,
            resize_inv_factor=ds_metainfo.resize_inv_factor)
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]

    def __call__(self, img):
        img = scale(img=img, size=self.resize_value)
        img = center_crop(img, self.input_image_size)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img


def calc_val_resize_value(input_image_size=(224, 224),
                          resize_inv_factor=0.875):
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))
    return resize_value
