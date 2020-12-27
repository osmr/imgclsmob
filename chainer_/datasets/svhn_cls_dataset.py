"""
    SVHN classification dataset.
"""

import os
from chainer.dataset import DatasetMixin
from chainer.datasets.svhn import get_svhn
from .cifar10_cls_dataset import CIFAR10MetaInfo


class SVHN(DatasetMixin):
    """
    SVHN image classification dataset from http://ufldl.stanford.edu/housenumbers/.
    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0`.

    Parameters:
    ----------
    root : str, default '~/.chainer/datasets/svhn'
        Path to temp folder for storing data.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".chainer", "datasets", "svhn"),
                 mode="train",
                 transform=None):
        assert (root is not None)
        self.transform = transform
        train_ds, test_ds = get_svhn()
        self.base = train_ds if mode == "train" else test_ds

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = self.transform(image)
        return image, label


class SVHNMetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(SVHNMetaInfo, self).__init__()
        self.label = "SVHN"
        self.root_dir_name = "svhn"
        self.dataset_class = SVHN
        self.num_training_samples = 73257
