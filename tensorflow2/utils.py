__all__ = ['load_image_imagenet1k_val', 'img_normalization', 'prepare_model']

import os
import cv2
import math
import logging
import numpy as np
from PIL import Image
import keras_preprocessing as keras_prep
import tensorflow as tf
from .tf2cv.model_provider import get_model


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


def img_normalization(img,
                      mean_rgb=(0.485, 0.456, 0.406),
                      std_rgb=(0.229, 0.224, 0.225)):
    """
    Normalization as in the ImageNet-1K validation procedure.

    Parameters
    ----------
    img : np.array
        input image.
    mean_rgb : tuple of 3 float
        Mean of RGB channels in the dataset.
    std_rgb : tuple of 3 float
        STD of RGB channels in the dataset.

    Returns
    -------
    np.array
        Output image.
    """
    mean_rgb = np.array(mean_rgb, np.float32) * 255.0
    std_rgb = np.array(std_rgb, np.float32) * 255.0
    img = (img - mean_rgb) / std_rgb
    return img


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


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  batch_size=None,
                  use_cuda=True):
    kwargs = {"pretrained": use_pretrained}
    # kwargs["input_shape"] = (1, 224, 224, 3)

    # my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    # tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
    # tf.debugging.set_log_device_placement(True)

    if not use_cuda:
        with tf.device("/cpu:0"):
            net = get_model(model_name, **kwargs)
            # input_shape = ((1, 3, net.in_size[0], net.in_size[1]) if
            #                net.data_format == "channels_first" else (1, net.in_size[0], net.in_size[1], 3))
            # net.build(input_shape=input_shape)
    else:
        net = get_model(model_name, **kwargs)
        # input_shape = ((batch_size, 3, net.in_size[0], net.in_size[1]) if
        #                net.data_format == "channels_first" else (batch_size, net.in_size[0], net.in_size[1], 3))
        # net.build(input_shape=input_shape)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))

        input_shape = ((batch_size, 3, net.in_size[0], net.in_size[1]) if
                       net.data_format == "channels_first" else (batch_size, net.in_size[0], net.in_size[1], 3))
        net.build(input_shape=input_shape)
        net.load_weights(filepath=pretrained_model_file_path)

    return net
