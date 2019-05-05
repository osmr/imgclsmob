"""
    CIFAR-100 classification dataset.
"""

from mxnet import gluon
from .cifar10_cls_dataset import CIFAR10MetaInfo


class CIFAR100Fine(gluon.data.vision.CIFAR100):

    def __init__(self,
                 root,
                 train):
        super(CIFAR100Fine, self).__init__(
            root=root,
            fine_label=True,
            train=train)


class CIFAR100MetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(CIFAR100MetaInfo, self).__init__()
        self.label = "CIFAR100"
        self.root_dir_name = "cifar100"
        self.dataset_class = CIFAR100Fine
        self.num_classes = 100
