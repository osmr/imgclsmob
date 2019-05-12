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
from .datasets.coco_seg_dataset import COCOMetaInfo
from mxnet.gluon.data import DataLoader
from mxnet.gluon.utils import split_and_load


def get_dataset_metainfo(dataset_name):
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
        "COCO": COCOMetaInfo,
    }
    if dataset_name in dataset_metainfo_map.keys():
        return dataset_metainfo_map[dataset_name]()
    else:
        raise Exception("Unrecognized dataset: {}".format(dataset_name))


def get_train_data_source(ds_metainfo,
                          batch_size,
                          num_workers):
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
            dataset = dataset.transform_first(fn=transform_train)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            last_batch="discard",
            num_workers=num_workers)


def get_val_data_source(ds_metainfo,
                        batch_size,
                        num_workers):
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
            dataset = dataset.transform_first(fn=transform_val)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)


def get_test_data_source(ds_metainfo,
                         batch_size,
                         num_workers):
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
            transform=(transform_test if ds_metainfo.do_transform else None))
        if not ds_metainfo.do_transform:
            dataset = dataset.transform_first(fn=transform_test)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)


def get_batch_fn(use_imgrec):
    if use_imgrec:
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