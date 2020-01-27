"""
    CIFAR-100 classification dataset.
"""

from tensorflow.keras.datasets import cifar100
from .cifar10_cls_dataset import CIFAR10MetaInfo


class CIFAR100MetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(CIFAR100MetaInfo, self).__init__()
        self.label = "CIFAR100"
        self.root_dir_name = "cifar100"
        self.num_classes = 100
        self.train_generator = cifar100_train_generator
        self.val_generator = cifar100_val_generator
        self.test_generator = cifar100_val_generator


def cifar100_train_generator(data_generator,
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
    assert(ds_metainfo is not None)
    (x_train, y_train), _ = cifar100.load_data()
    generator = data_generator.flow(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=False)
    return generator


def cifar100_val_generator(data_generator,
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
    assert(ds_metainfo is not None)
    _, (x_test, y_test) = cifar100.load_data()
    generator = data_generator.flow(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        shuffle=False)
    return generator
