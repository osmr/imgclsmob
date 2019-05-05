"""
    ImageNet-1K/CUB-200-2011 dataset routines.
"""

__all__ = ['get_train_data_source', 'get_val_data_source']

import os
from mxnet import gluon
from gluon.datasets.imagenet1k_cls_dataset import calc_val_resize_value, imagenet_val_transform,\
    imagenet_train_transform
from gluon.datasets.imagenet1k_rec_cls_dataset import imagenet_train_imgrec_iter, imagenet_val_imgrec_iter


def get_train_data_source(dataset_metainfo,
                          dataset_dir,
                          batch_size,
                          num_workers,
                          input_image_size=(224, 224)):
    if dataset_metainfo.use_imgrec:
        if isinstance(input_image_size, int):
            input_image_size = (input_image_size, input_image_size)
        data_shape = (3,) + input_image_size
        return imagenet_train_imgrec_iter(
            imgrec_file_path=os.path.join(dataset_dir, dataset_metainfo.train_imgrec_file_path),
            imgidx_file_path=os.path.join(dataset_dir, dataset_metainfo.train_imgidx_file_path),
            batch_size=batch_size,
            num_workers=num_workers,
            data_shape=data_shape)
    else:
        dataset_class = dataset_metainfo.dataset_class
        transform_train = imagenet_train_transform(input_image_size=input_image_size)
        dataset = dataset_class(
            root=dataset_dir,
            train=True).transform_first(fn=transform_train)
        return gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            last_batch="discard",
            num_workers=num_workers)


def get_val_data_source(dataset_metainfo,
                        dataset_dir,
                        batch_size,
                        num_workers,
                        input_image_size=(224, 224),
                        resize_inv_factor=0.875):
    resize_value = calc_val_resize_value(
        input_image_size=input_image_size,
        resize_inv_factor=resize_inv_factor)
    if dataset_metainfo.use_imgrec:
        if isinstance(input_image_size, int):
            input_image_size = (input_image_size, input_image_size)
        data_shape = (3,) + input_image_size
        return imagenet_val_imgrec_iter(
            imgrec_file_path=os.path.join(dataset_dir, dataset_metainfo.val_imgrec_file_path),
            imgidx_file_path=os.path.join(dataset_dir, dataset_metainfo.val_imgidx_file_path),
            batch_size=batch_size,
            num_workers=num_workers,
            data_shape=data_shape,
            resize_value=resize_value)
    else:
        dataset_class = dataset_metainfo.dataset_class
        transform_val = imagenet_val_transform(
            input_image_size=input_image_size,
            resize_value=resize_value)
        dataset = dataset_class(
            root=dataset_dir,
            train=False).transform_first(fn=transform_val)
        return gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
