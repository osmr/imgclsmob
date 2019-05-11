"""
    SVHN classification dataset.
"""

import os
from torchvision.datasets import SVHN
from .cifar10_cls_dataset import CIFAR10MetaInfo


class SVHNFine(SVHN):
    """
    SVHN image classification dataset from http://ufldl.stanford.edu/housenumbers/.
    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0`.

    Parameters
    ----------
    root : str, default '~/.torch/datasets/svhn'
        Path to temp folder for storing data.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".torch", "datasets", "svhn"),
                 mode="train",
                 transform=None):
        super(SVHNFine, self).__init__(
            root=root,
            split=("train" if mode == "train" else "test"),
            transform=transform,
            download=True)


class SVHNMetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(SVHNMetaInfo, self).__init__()
        self.label = "SVHN"
        self.root_dir_name = "svhn"
        self.dataset_class = SVHN
        self.num_training_samples = 73257
