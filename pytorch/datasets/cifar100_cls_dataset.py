"""
    CIFAR-100 classification dataset.
"""

import os
from torchvision.datasets import CIFAR100
from .cifar10_cls_dataset import CIFAR10MetaInfo


class CIFAR100Fine(CIFAR100):
    """
    CIFAR-100 image classification dataset.


    Parameters
    ----------
    root : str, default '~/.torch/datasets/cifar100'
        Path to temp folder for storing data.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".torch", "datasets", "cifar100"),
                 mode="train",
                 transform=None):
        super(CIFAR100Fine, self).__init__(
            root=root,
            train=(mode == "train"),
            transform=transform,
            download=True)


class CIFAR100MetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(CIFAR100MetaInfo, self).__init__()
        self.label = "CIFAR100"
        self.root_dir_name = "cifar100"
        self.dataset_class = CIFAR100Fine
        self.num_classes = 100
