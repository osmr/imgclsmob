"""
    CIFAR-10 classification dataset.
"""

import os
import numpy as np
from chainer.dataset import DatasetMixin
from chainer.datasets.cifar import get_cifar10
from .dataset_metainfo import DatasetMetaInfo


class CIFAR10Fine(DatasetMixin):
    """
    CIFAR-10 image classification dataset.


    Parameters
    ----------
    root : str, default '~/.chainer/datasets/cifar10'
        Path to temp folder for storing data.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".chainer", "datasets", "cifar10"),
                 mode="train",
                 transform=None):
        assert (root is not None)
        self.transform = transform
        train_ds, test_ds = get_cifar10()
        self.base = train_ds if mode == "train" else test_ds

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = self.transform(image)
        return image, label


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
        self.train_transform = None
        self.val_transform = CIFARValTransform
        self.test_transform = CIFARValTransform
        self.ml_type = "imgcls"


class CIFARValTransform(object):
    """
    CIFAR-10 validation transform.
    """
    def __init__(self,
                 ds_metainfo,
                 mean_rgb=(0.4914, 0.4822, 0.4465),
                 std_rgb=(0.2023, 0.1994, 0.2010)):
        assert (ds_metainfo is not None)
        self.mean = np.array(mean_rgb, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std_rgb, np.float32)[:, np.newaxis, np.newaxis]

    def __call__(self, img):
        img -= self.mean
        img /= self.std
        return img
