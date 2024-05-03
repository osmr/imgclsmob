"""
    CIFAR-100 classification dataset.
"""

import os
from chainer.dataset import DatasetMixin
from chainer.datasets.cifar import get_cifar100
from .cifar10_cls_dataset import CIFAR10MetaInfo


class CIFAR100(DatasetMixin):
    """
    CIFAR-100 image classification dataset.


    Parameters
    ----------
    root : str, default '~/.chainer/datasets/cifar100'
        Path to temp folder for storing data.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".chainer", "datasets", "cifar100"),
                 mode="train",
                 transform=None):
        assert (root is not None)
        self.transform = transform
        train_ds, test_ds = get_cifar100()
        self.base = train_ds if mode == "train" else test_ds

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = self.transform(image)
        return image, label


class CIFAR100MetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(CIFAR100MetaInfo, self).__init__()
        self.label = "CIFAR100"
        self.root_dir_name = "cifar100"
        self.dataset_class = CIFAR100
        self.num_classes = 100
