"""
    Dataset routines.
"""

__all__ = ['get_dataset_metainfo', 'get_train_data_source', 'get_val_data_source', 'get_test_data_source',
           'get_batch_fn']

from .datasets.imagenet1k_cls_dataset import ImageNet1KMetaInfo
from .datasets.imagenet1k_rec_cls_dataset import ImageNet1KRecMetaInfo
from .datasets.cub200_2011_cls_dataset import CUB200MetaInfo
from .datasets.cifar10_cls_dataset import CIFAR10MetaInfo
from .datasets.cifar100_cls_dataset import CIFAR100MetaInfo
from .datasets.svhn_cls_dataset import SVHNMetaInfo
from .datasets.voc_seg_dataset import VOCMetaInfo
from .datasets.ade20k_seg_dataset import ADE20KMetaInfo
from .datasets.cityscapes_seg_dataset import CityscapesMetaInfo
from .datasets.coco_seg_dataset import CocoSegMetaInfo
from .datasets.coco_det_dataset import CocoDetMetaInfo
from .datasets.widerface_det_dataset import WiderfaceDetMetaInfo
from .datasets.coco_hpe1_dataset import CocoHpe1MetaInfo
from .datasets.coco_hpe2_dataset import CocoHpe2MetaInfo
from .datasets.coco_hpe3_dataset import CocoHpe3MetaInfo
from .datasets.hpatches_mch_dataset import HPatchesMetaInfo
from .weighted_random_sampler import WeightedRandomSampler
from mxnet.gluon.data import DataLoader
from mxnet.gluon.utils import split_and_load


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
        "ImageNet1K_rec": ImageNet1KRecMetaInfo,
        "CUB200_2011": CUB200MetaInfo,
        "CIFAR10": CIFAR10MetaInfo,
        "CIFAR100": CIFAR100MetaInfo,
        "SVHN": SVHNMetaInfo,
        "VOC": VOCMetaInfo,
        "ADE20K": ADE20KMetaInfo,
        "Cityscapes": CityscapesMetaInfo,
        "CocoSeg": CocoSegMetaInfo,
        "CocoDet": CocoDetMetaInfo,
        "WiderFace": WiderfaceDetMetaInfo,
        "CocoHpe1": CocoHpe1MetaInfo,
        "CocoHpe2": CocoHpe2MetaInfo,
        "CocoHpe3": CocoHpe3MetaInfo,
        "HPatches": HPatchesMetaInfo,
    }
    if dataset_name in dataset_metainfo_map.keys():
        return dataset_metainfo_map[dataset_name]()
    else:
        raise Exception("Unrecognized dataset: {}".format(dataset_name))


def get_train_data_source(ds_metainfo,
                          batch_size,
                          num_workers):
    """
    Get data source for training subset.

    Parameters
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns
    -------
    DataLoader or ImageRecordIter
        Data source.
    """
    if ds_metainfo.use_imgrec:
        return ds_metainfo.train_imgrec_iter(
            ds_metainfo=ds_metainfo,
            batch_size=batch_size,
            num_workers=num_workers)
    else:
        transform_train = ds_metainfo.train_transform(ds_metainfo=ds_metainfo)
        dataset = ds_metainfo.dataset_class(
            root=ds_metainfo.root_dir_path,
            mode="train",
            transform=(transform_train if ds_metainfo.do_transform else None))
        if not ds_metainfo.do_transform:
            if ds_metainfo.do_transform_first:
                dataset = dataset.transform_first(fn=transform_train)
            else:
                dataset = dataset.transform(fn=transform_train)
        ds_metainfo.update_from_dataset(dataset)
        if not ds_metainfo.train_use_weighted_sampler:
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                last_batch="discard",
                num_workers=num_workers)
        else:
            sampler = WeightedRandomSampler(
                length=len(dataset),
                weights=dataset._data.sample_weights)
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                # shuffle=True,
                sampler=sampler,
                last_batch="discard",
                batchify_fn=ds_metainfo.batchify_fn,
                num_workers=num_workers)


def get_val_data_source(ds_metainfo,
                        batch_size,
                        num_workers):
    """
    Get data source for validation subset.

    Parameters
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns
    -------
    DataLoader or ImageRecordIter
        Data source.
    """
    if ds_metainfo.use_imgrec:
        return ds_metainfo.val_imgrec_iter(
            ds_metainfo=ds_metainfo,
            batch_size=batch_size,
            num_workers=num_workers)
    else:
        transform_val = ds_metainfo.val_transform(ds_metainfo=ds_metainfo)
        dataset = ds_metainfo.dataset_class(
            root=ds_metainfo.root_dir_path,
            mode="val",
            transform=(transform_val if ds_metainfo.do_transform else None))
        if not ds_metainfo.do_transform:
            if ds_metainfo.do_transform_first:
                dataset = dataset.transform_first(fn=transform_val)
            else:
                dataset = dataset.transform(fn=transform_val)
        ds_metainfo.update_from_dataset(dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            last_batch=ds_metainfo.batchify_fn,
            batchify_fn=ds_metainfo.batchify_fn,
            num_workers=num_workers)


def get_test_data_source(ds_metainfo,
                         batch_size,
                         num_workers):
    """
    Get data source for testing subset.

    Parameters
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns
    -------
    DataLoader or ImageRecordIter
        Data source.
    """
    if ds_metainfo.use_imgrec:
        return ds_metainfo.val_imgrec_iter(
            ds_metainfo=ds_metainfo,
            batch_size=batch_size,
            num_workers=num_workers)
    else:
        transform_test = ds_metainfo.test_transform(ds_metainfo=ds_metainfo)
        dataset = ds_metainfo.dataset_class(
            root=ds_metainfo.root_dir_path,
            mode="test",
            transform=(transform_test if ds_metainfo.do_transform else None),
            **ds_metainfo.test_dataset_extra_kwargs)
        if not ds_metainfo.do_transform:
            if ds_metainfo.do_transform_first:
                dataset = dataset.transform_first(fn=transform_test)
            else:
                dataset = dataset.transform(fn=transform_test)
        ds_metainfo.update_from_dataset(dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            last_batch=ds_metainfo.last_batch,
            batchify_fn=ds_metainfo.batchify_fn,
            num_workers=num_workers)


def get_batch_fn(ds_metainfo):
    """
    Get function for splitting data after extraction from data loader.

    Parameters
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.

    Returns
    -------
    func
        Desired function.
    """
    if ds_metainfo.use_imgrec:
        def batch_fn(batch, ctx):
            data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            return data, label

        return batch_fn
    else:
        def batch_fn(batch, ctx):
            data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label

        return batch_fn
