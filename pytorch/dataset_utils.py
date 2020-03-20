"""
    Dataset routines.
"""

__all__ = ['get_dataset_metainfo', 'get_train_data_source', 'get_val_data_source', 'get_test_data_source']

from .datasets.imagenet1k_cls_dataset import ImageNet1KMetaInfo
from .datasets.cub200_2011_cls_dataset import CUB200MetaInfo
from .datasets.cifar10_cls_dataset import CIFAR10MetaInfo
from .datasets.cifar100_cls_dataset import CIFAR100MetaInfo
from .datasets.svhn_cls_dataset import SVHNMetaInfo
from .datasets.voc_seg_dataset import VOCMetaInfo
from .datasets.ade20k_seg_dataset import ADE20KMetaInfo
from .datasets.cityscapes_seg_dataset import CityscapesMetaInfo
from .datasets.coco_seg_dataset import CocoSegMetaInfo
from .datasets.coco_det_dataset import CocoDetMetaInfo
from .datasets.coco_hpe1_dataset import CocoHpe1MetaInfo
from .datasets.coco_hpe2_dataset import CocoHpe2MetaInfo
from .datasets.coco_hpe3_dataset import CocoHpe3MetaInfo
from .datasets.hpatches_mch_dataset import HPatchesMetaInfo
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


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
        "CocoDet": CocoDetMetaInfo,
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
    DataLoader
        Data source.
    """
    transform_train = ds_metainfo.train_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="train",
        transform=transform_train,
        **kwargs)
    ds_metainfo.update_from_dataset(dataset)
    if not ds_metainfo.train_use_weighted_sampler:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
    else:
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset))
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True)


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
    DataLoader
        Data source.
    """
    transform_val = ds_metainfo.val_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="val",
        transform=transform_val,
        **kwargs)
    ds_metainfo.update_from_dataset(dataset)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)


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
    DataLoader
        Data source.
    """
    transform_test = ds_metainfo.test_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="test",
        transform=transform_test,
        **kwargs)
    ds_metainfo.update_from_dataset(dataset)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
