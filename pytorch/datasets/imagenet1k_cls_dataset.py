"""
    ImageNet-1K classification dataset.
"""

import os
import math
import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from .dataset_metainfo import DatasetMetaInfo


class ImageNet1K(ImageFolder):
    """
    ImageNet-1K classification dataset.

    Parameters
    ----------
    root : str, default '~/.torch/datasets/imagenet'
        Path to the folder stored the dataset.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".torch", "datasets", "imagenet"),
                 mode="train",
                 transform=None):
        split = "train" if mode == "train" else "val"
        root = os.path.join(root, split)
        super(ImageNet1K, self).__init__(root=root, transform=transform)


class ImageNet1KMetaInfo(DatasetMetaInfo):
    """
    Descriptor of ImageNet-1K dataset.
    """

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
        self.use_cv_resize = False
        self.mean_rgb = (0.485, 0.456, 0.406)
        self.std_rgb = (0.229, 0.224, 0.225)
        self.interpolation = Image.BILINEAR

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
        parser.add_argument(
            "--use-cv-resize",
            action="store_true",
            help="use OpenCV resize preprocessing")
        parser.add_argument(
            "--mean-rgb",
            nargs=3,
            type=float,
            default=self.mean_rgb,
            help="Mean of RGB channels in the dataset")
        parser.add_argument(
            "--std-rgb",
            nargs=3,
            type=float,
            default=self.std_rgb,
            help="STD of RGB channels in the dataset")
        parser.add_argument(
            "--interpolation",
            type=int,
            default=self.interpolation,
            help="Preprocessing interpolation")

    def update(self,
               args):
        """
        Update ImageNet-1K dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        super(ImageNet1KMetaInfo, self).update(args)
        self.input_image_size = (args.input_size, args.input_size)
        self.use_cv_resize = args.use_cv_resize
        self.mean_rgb = args.mean_rgb
        self.std_rgb = args.std_rgb
        self.interpolation = args.interpolation


def imagenet_train_transform(ds_metainfo,
                             jitter_param=0.4):
    """
    Create image transform sequence for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    jitter_param : float
        How much to jitter values.

    Returns
    -------
    Compose
        Image transform sequence.
    """
    input_image_size = ds_metainfo.input_image_size
    return transforms.Compose([
        transforms.RandomResizedCrop(size=input_image_size, interpolation=ds_metainfo.interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ds_metainfo.mean_rgb,
            std=ds_metainfo.std_rgb)
    ])


def imagenet_val_transform(ds_metainfo):
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.

    Returns
    -------
    Compose
        Image transform sequence.
    """
    input_image_size = ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    return transforms.Compose([
        CvResize(size=resize_value, interpolation=ds_metainfo.interpolation) if ds_metainfo.use_cv_resize else
        transforms.Resize(size=resize_value, interpolation=ds_metainfo.interpolation),
        transforms.CenterCrop(size=input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ds_metainfo.mean_rgb,
            std=ds_metainfo.std_rgb)
    ])


class CvResize(object):
    """
    Resize the input PIL Image to the given size via OpenCV.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    interpolation : int, default PIL.Image.BILINEAR
        Interpolation method for resizing. By default uses bilinear
        interpolation.
    """
    def __init__(self,
                 size,
                 interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Resize image.

        Parameters
        ----------
        img : PIL.Image
            input image.

        Returns
        -------
        PIL.Image
            Resulted image.
        """
        if self.interpolation == Image.NEAREST:
            cv_interpolation = cv2.INTER_NEAREST
        elif self.interpolation == Image.BILINEAR:
            cv_interpolation = cv2.INTER_LINEAR
        elif self.interpolation == Image.BICUBIC:
            cv_interpolation = cv2.INTER_CUBIC
        elif self.interpolation == Image.LANCZOS:
            cv_interpolation = cv2.INTER_LANCZOS4
        else:
            raise ValueError()

        cv_img = np.array(img)

        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                out_size = (self.size, int(self.size * h / w))
            else:
                out_size = (int(self.size * w / h), self.size)
            cv_img = cv2.resize(cv_img, dsize=out_size, interpolation=cv_interpolation)
            return Image.fromarray(cv_img)
        else:
            cv_img = cv2.resize(cv_img, dsize=self.size, interpolation=cv_interpolation)
            return Image.fromarray(cv_img)


def calc_val_resize_value(input_image_size=(224, 224),
                          resize_inv_factor=0.875):
    """
    Calculate image resize value for validation subset.

    Parameters:
    ----------
    input_image_size : tuple of 2 int
        Main script arguments.
    resize_inv_factor : float
        Resize inverted factor.

    Returns
    -------
    int
        Resize value.
    """
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))
    return resize_value
