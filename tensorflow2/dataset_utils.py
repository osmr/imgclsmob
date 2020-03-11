"""
    Dataset routines.
"""

__all__ = ['get_dataset_metainfo', 'get_train_data_source', 'get_val_data_source', 'get_test_data_source']

import tensorflow as tf
from .datasets.imagenet1k_cls_dataset import ImageNet1KMetaInfo
from .datasets.cub200_2011_cls_dataset import CUB200MetaInfo
from .datasets.cifar10_cls_dataset import CIFAR10MetaInfo
from .datasets.cifar100_cls_dataset import CIFAR100MetaInfo
from .datasets.svhn_cls_dataset import SVHNMetaInfo
from .datasets.voc_seg_dataset import VOCMetaInfo
from .datasets.ade20k_seg_dataset import ADE20KMetaInfo
from .datasets.cityscapes_seg_dataset import CityscapesMetaInfo
from .datasets.coco_seg_dataset import CocoSegMetaInfo
from .datasets.coco_hpe1_dataset import CocoHpe1MetaInfo
from .datasets.coco_hpe2_dataset import CocoHpe2MetaInfo
from .datasets.coco_hpe3_dataset import CocoHpe3MetaInfo


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
        "CUB200_2011": CUB200MetaInfo,
        "CIFAR10": CIFAR10MetaInfo,
        "CIFAR100": CIFAR100MetaInfo,
        "SVHN": SVHNMetaInfo,
        "VOC": VOCMetaInfo,
        "ADE20K": ADE20KMetaInfo,
        "Cityscapes": CityscapesMetaInfo,
        "CocoSeg": CocoSegMetaInfo,
        "CocoHpe1": CocoHpe1MetaInfo,
        "CocoHpe2": CocoHpe2MetaInfo,
        "CocoHpe3": CocoHpe3MetaInfo,
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
    if hasattr(generator, "dataset"):
        ds_metainfo.update_from_dataset(generator.dataset)
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
    data_generator = ds_metainfo.test_transform(
        ds_metainfo=ds_metainfo,
        data_format=data_format)
    generator = ds_metainfo.test_generator(
        data_generator=data_generator,
        ds_metainfo=ds_metainfo,
        batch_size=batch_size)
    if hasattr(generator, "dataset"):
        ds_metainfo.update_from_dataset(generator.dataset)
    return tf.data.Dataset.from_generator(
        generator=lambda: generator,
        output_types=(tf.float32, tf.float32)),\
           generator.n
