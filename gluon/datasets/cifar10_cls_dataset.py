"""
    CIFAR-10 classification dataset.
"""

import os
import numpy as np
import mxnet as mx
from mxnet.gluon import Block
from mxnet.gluon.data.vision import CIFAR10
from mxnet.gluon.data.vision import transforms
from .dataset_metainfo import DatasetMetaInfo


class CIFAR10Fine(CIFAR10):
    """
    CIFAR-10 image classification dataset.


    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar10
        Path to temp folder for storing data.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A user defined callback that transforms each sample.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "cifar10"),
                 mode="train",
                 transform=None):
        super(CIFAR10Fine, self).__init__(
            root=root,
            train=(mode == "train"),
            transform=transform)


class CIFAR10MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CIFAR10MetaInfo, self).__init__()
        self.label = "CIFAR10"
        self.short_label = "cifar"
        self.root_dir_name = "cifar10"
        self.dataset_class = CIFAR10Fine
        self.num_training_samples = 50000
        self.in_channels = 3
        self.num_classes = 10
        self.input_image_size = (32, 32)
        self.train_metric_capts = ["Train.Err"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err"}]
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.saver_acc_ind = 0
        self.train_transform = cifar10_train_transform
        self.val_transform = cifar10_val_transform
        self.test_transform = cifar10_val_transform
        self.ml_type = "imgcls"
        self.loss_name = "SoftmaxCrossEntropy"


class RandomCrop(Block):
    """
    Randomly crop `src` with `size` (width, height).
    Padding is optional.
    Upsample result if `src` is smaller than `size`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of the final output.
    pad: int or tuple, default None
        if int, size of the zero-padding
        if tuple, number of values padded to the edges of each axis.
            ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
            ((before, after),) yields same before and after pad for each axis.
            (pad,) or int is a shortcut for before = after = pad width for all axes.
    interpolation : int, default 2
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
    """
    def __init__(self,
                 size,
                 pad=None,
                 interpolation=2):
        super(RandomCrop, self).__init__()
        numeric_types = (float, int, np.generic)
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, interpolation)
        if isinstance(pad, int):
            self.pad = ((pad, pad), (pad, pad), (0, 0))
        else:
            self.pad = pad

    def forward(self, x):
        if self.pad:
            x_pad = np.pad(x.asnumpy(), self.pad, mode="constant", constant_values=0)
        return mx.image.random_crop(mx.nd.array(x_pad), *self._args)[0]


def cifar10_train_transform(ds_metainfo,
                            mean_rgb=(0.4914, 0.4822, 0.4465),
                            std_rgb=(0.2023, 0.1994, 0.2010),
                            jitter_param=0.4,
                            lighting_param=0.1):
    assert (ds_metainfo is not None)
    assert (ds_metainfo.input_image_size[0] == 32)
    return transforms.Compose([
        RandomCrop(
            size=32,
            pad=4),
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


def cifar10_val_transform(ds_metainfo,
                          mean_rgb=(0.4914, 0.4822, 0.4465),
                          std_rgb=(0.2023, 0.1994, 0.2010)):
    assert (ds_metainfo is not None)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
