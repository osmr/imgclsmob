"""
    CIFAR/SVHN dataset routines.
"""

__all__ = ['get_train_data_source', 'get_val_data_source']

from mxnet import gluon
from gluon.datasets.cifar10_cls_dataset import cifar10_train_transform, cifar10_val_transform


def get_train_data_source(ds_metainfo,
                          dataset_dir,
                          batch_size,
                          num_workers):
    dataset_class = ds_metainfo.dataset_class
    transform_train = cifar10_train_transform(ds_metainfo=ds_metainfo)
    dataset = dataset_class(
        root=dataset_dir,
        train=True).transform_first(fn=transform_train)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        last_batch="discard",
        num_workers=num_workers)


def get_val_data_source(ds_metainfo,
                        dataset_dir,
                        batch_size,
                        num_workers):
    dataset_class = ds_metainfo.dataset_class
    transform_val = cifar10_val_transform(ds_metainfo=ds_metainfo)
    dataset = dataset_class(
        root=dataset_dir,
        train=False).transform_first(fn=transform_val)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
