"""
    ImageNet-1K classification dataset.
"""

import os
import math
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data.vision import transforms


class ImageNet(ImageFolderDataset):
    """
    Load the ImageNet classification dataset.

    Refer to :doc:`../build/examples_datasets/imagenet` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "imagenet"),
                 train=True,
                 transform=None):
        split = "train" if train else "val"
        root = os.path.join(root, split)
        super(ImageNet, self).__init__(root=root, flag=1, transform=transform)


def imagenet_train_transform(input_image_size=(224, 224),
                             mean_rgb=(0.485, 0.456, 0.406),
                             std_rgb=(0.229, 0.224, 0.225),
                             jitter_param=0.4,
                             lighting_param=0.1):
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


def imagenet_val_transform(input_image_size=(224, 224),
                           mean_rgb=(0.485, 0.456, 0.406),
                           std_rgb=(0.229, 0.224, 0.225),
                           resize_value=256):
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
