"""
    ImageNet-1K classification dataset.
"""

__all__ = ['ImageNet1KMetaInfo', 'load_image_imagenet1k_val']

import os
import math
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_preprocessing as keras_prep
from .dataset_metainfo import DatasetMetaInfo
from .cls_dataset import img_normalization


class ImageNet1KMetaInfo(DatasetMetaInfo):
    """
    Descriptor of ImageNet-1K dataset.
    """

    def __init__(self):
        super(ImageNet1KMetaInfo, self).__init__()
        self.label = "ImageNet1K"
        self.short_label = "imagenet"
        self.root_dir_name = "imagenet"
        self.dataset_class = None
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
        self.train_generator = imagenet_train_generator
        self.val_generator = imagenet_val_generator
        self.test_generator = imagenet_val_generator
        self.ml_type = "imgcls"
        self.mean_rgb = (0.485, 0.456, 0.406)
        self.std_rgb = (0.229, 0.224, 0.225)
        self.interpolation = "bilinear"
        self.interpolation_msg = "bilinear"

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
            type=str,
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
        self.mean_rgb = args.mean_rgb
        self.std_rgb = args.std_rgb
        self.interpolation = args.interpolation

        if self.interpolation == "nearest":
            self.interpolation_msg = self.interpolation
        else:
            self.interpolation_msg = "{}:{}".format(self.interpolation, self.resize_inv_factor)
            import keras_preprocessing as keras_prep
            keras_prep.image.iterator.load_img = load_image_imagenet1k_val


def resize(img,
           size,
           interpolation):
    """
    Resize the input PIL Image to the given size via OpenCV.

    Parameters
    ----------
    img : PIL.Image
        input image.
    size : int or tuple of (W, H)
        Size of output image.
    interpolation : int
        Interpolation method for resizing.

    Returns
    -------
    PIL.Image
        Resulted image.
    """
    if interpolation == Image.NEAREST:
        cv_interpolation = cv2.INTER_NEAREST
    elif interpolation == Image.BILINEAR:
        cv_interpolation = cv2.INTER_LINEAR
    elif interpolation == Image.BICUBIC:
        cv_interpolation = cv2.INTER_CUBIC
    elif interpolation == Image.LANCZOS:
        cv_interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError("Invalid interpolation method: {}", interpolation)

    cv_img = np.array(img)

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            out_size = (size, int(size * h / w))
        else:
            out_size = (int(size * w / h), size)
        cv_img = cv2.resize(cv_img, dsize=out_size, interpolation=cv_interpolation)
        return Image.fromarray(cv_img)
    else:
        cv_img = cv2.resize(cv_img, dsize=size, interpolation=cv_interpolation)
        return Image.fromarray(cv_img)


def center_crop(img,
                output_size):
    """
    Crop the given PIL Image.

    Parameters
    ----------
    img : PIL.Image
        input image.
    output_size : tuple of (W, H)
        Size of output image.

    Returns
    -------
    PIL.Image
        Resulted image.
    """
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img.crop((j, i, j + tw, i + th))


def load_image_imagenet1k_val(path,
                              grayscale=False,
                              color_mode="rgb",
                              target_size=None,
                              interpolation="nearest"):
    """
    Wraps keras_preprocessing.image.utils.load_img and apply center crop as in ImageNet-1K validation procedure.

    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", 'rgb', 'rgba'. Default: 'rgb'.
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is an inverted ratio for input
            image crop, e.g. 'lanczos:0.875'.
            Supported interpolation methods are 'nearest', 'bilinear', 'bicubic', 'lanczos',
            'box', 'hamming' By default, 'nearest' is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    interpolation, resize_inv_factor = interpolation.split(":") if ":" in interpolation else (interpolation, "none")
    if resize_inv_factor == "none":
        return keras_prep.image.utils.load_img(
            path=path,
            grayscale=grayscale,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation)

    img = keras_prep.image.utils.load_img(
        path=path,
        grayscale=grayscale,
        color_mode=color_mode,
        target_size=None,
        interpolation=interpolation)

    if (target_size is None) or (img.size == (target_size[1], target_size[0])):
        return img

    try:
        resize_inv_factor = float(resize_inv_factor)
    except ValueError:
        raise ValueError("Invalid crop inverted ratio: {}", resize_inv_factor)

    if interpolation not in keras_prep.image.utils._PIL_INTERPOLATION_METHODS:
        raise ValueError("Invalid interpolation method {} specified. Supported methods are {}".format(
            interpolation,
            ", ".join(keras_prep.image.utils._PIL_INTERPOLATION_METHODS.keys())))
    resample = keras_prep.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

    resize_value = int(math.ceil(float(target_size[0]) / resize_inv_factor))

    img = resize(
        img=img,
        size=resize_value,
        interpolation=resample)
    return center_crop(
        img=img,
        output_size=target_size)


def imagenet_train_transform(ds_metainfo,
                             data_format="channels_last"):
    """
    Create image transform sequence for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = ImageDataGenerator(
        preprocessing_function=(lambda img: img_normalization(
            img=img,
            mean_rgb=ds_metainfo.mean_rgb,
            std_rgb=ds_metainfo.std_rgb)),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format=data_format)
    return data_generator


def imagenet_val_transform(ds_metainfo,
                           data_format="channels_last"):
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = ImageDataGenerator(
        preprocessing_function=(lambda img: img_normalization(
            img=img,
            mean_rgb=ds_metainfo.mean_rgb,
            std_rgb=ds_metainfo.std_rgb)),
        data_format=data_format)
    return data_generator


def imagenet_train_generator(data_generator,
                             ds_metainfo,
                             batch_size):
    """
    Create image generator for training subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    split = "train"
    root = ds_metainfo.root_dir_path
    root = os.path.join(root, split)
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation=ds_metainfo.interpolation_msg)
    return generator


def imagenet_val_generator(data_generator,
                           ds_metainfo,
                           batch_size):
    """
    Create image generator for validation subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    split = "val"
    root = ds_metainfo.root_dir_path
    root = os.path.join(root, split)
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation=ds_metainfo.interpolation_msg)
    return generator
