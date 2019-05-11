"""
    ImageNet-1K classification dataset.
"""

import os
import math
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data.vision import transforms
from .dataset_metainfo import DatasetMetaInfo


class ImageNet1K(ImageFolderDataset):
    """
    ImageNet-1K classification dataset.

    Refer to MXNet documentation for the description of this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "imagenet"),
                 mode="train",
                 transform=None):
        split = "train" if mode == "train" else "val"
        root = os.path.join(root, split)
        super(ImageNet1K, self).__init__(root=root, flag=1, transform=transform)


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
        self.train_transform = imagenet_train_transform
        self.val_transform = imagenet_val_transform
        self.test_transform = imagenet_val_transform
        self.ml_type = "imgcls"

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
        self.resize_inv_factor = args.resize_inv_factor


def imagenet_train_transform(ds_metainfo,
                             mean_rgb=(0.485, 0.456, 0.406),
                             std_rgb=(0.229, 0.224, 0.225),
                             jitter_param=0.4,
                             lighting_param=0.1):
    input_image_size = ds_metainfo.input_image_size
    return transforms.Compose([
        transforms.RandomResizedCrop(input_image_size),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])


def imagenet_val_transform(ds_metainfo,
                           mean_rgb=(0.485, 0.456, 0.406),
                           std_rgb=(0.229, 0.224, 0.225)):
    input_image_size = ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    return transforms.Compose([
        transforms.Resize(
            size=resize_value,
            keep_ratio=True),
        transforms.CenterCrop(size=input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])


def calc_val_resize_value(input_image_size=(224, 224),
                          resize_inv_factor=0.875):
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))
    return resize_value
