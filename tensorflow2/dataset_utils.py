"""
    Dataset routines.
"""

__all__ = ['get_dataset_metainfo', 'get_train_data_source', 'get_val_data_source', 'get_test_data_source']

import tensorflow as tf
from .datasets.imagenet1k_cls_dataset import ImageNet1KMetaInfo
from .datasets.cifar10_cls_dataset import CIFAR10MetaInfo
from .datasets.cifar100_cls_dataset import CIFAR100MetaInfo
from .datasets.svhn_cls_dataset import SVHNMetaInfo


def get_dataset_metainfo(dataset_name):
    """
    Get dataset metainfo by name of dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name.

    Returns
    -------
    DatasetMetaInfo
        Dataset metainfo.
    """
    dataset_metainfo_map = {
        "ImageNet1K": ImageNet1KMetaInfo,
        "CIFAR10": CIFAR10MetaInfo,
        "CIFAR100": CIFAR100MetaInfo,
        "SVHN": SVHNMetaInfo,
    }
    if dataset_name in dataset_metainfo_map.keys():
        return dataset_metainfo_map[dataset_name]()
    else:
        raise Exception("Unrecognized dataset: {}".format(dataset_name))


def get_train_data_source(ds_metainfo,
                          batch_size,
                          data_format="channels_last"):
    """
    Get data source for training subset.

    Parameters
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    DataLoader
        Data source.
    int
        Dataset size.
    """
    data_generator = ds_metainfo.train_transform(
        ds_metainfo=ds_metainfo,
        data_format=data_format)
    generator = ds_metainfo.train_generator(
        data_generator=data_generator,
        ds_metainfo=ds_metainfo,
        batch_size=batch_size)
    return tf.data.Dataset.from_generator(
        generator=lambda: generator,
        output_types=(tf.float32, tf.float32)),\
           generator.n


def get_val_data_source(ds_metainfo,
                        batch_size,
                        data_format="channels_last"):
    """
    Get data source for validation subset.

    Parameters
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    DataLoader
        Data source.
    int
        Dataset size.
    """
    data_generator = ds_metainfo.val_transform(
        ds_metainfo=ds_metainfo,
        data_format=data_format)
    generator = ds_metainfo.val_generator(
        data_generator=data_generator,
        ds_metainfo=ds_metainfo,
        batch_size=batch_size)
    return tf.data.Dataset.from_generator(
        generator=lambda: generator,
        output_types=(tf.float32, tf.float32)),\
           generator.n


def get_test_data_source(ds_metainfo,
                         batch_size,
                         data_format="channels_last"):
    """
    Get data source for testing subset.

    Parameters
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    DataLoader
        Data source.
    int
        Dataset size.
    """
    data_generator = ds_metainfo.val_transform(
        ds_metainfo=ds_metainfo,
        data_format=data_format)
    generator = ds_metainfo.val_generator(
        data_generator=data_generator,
        ds_metainfo=ds_metainfo,
        batch_size=batch_size)
    return tf.data.Dataset.from_generator(
        generator=lambda: generator,
        output_types=(tf.float32, tf.float32)),\
           generator.n
