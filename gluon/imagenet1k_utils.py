"""
    ImageNet-1K dataset routines.
"""

__all__ = ['get_dataset_metainfo', 'add_dataset_parser_arguments', 'get_batch_fn', 'get_train_data_source',
           'get_val_data_source']

import os
from mxnet import gluon
from gluon.datasets.imagenet1k_cls_dataset import ImageNet1KMetaInfo, ImageNet1K, calc_val_resize_value,\
    imagenet_val_transform, imagenet_train_transform
from gluon.datasets.imagenet1k_rec_cls_dataset import ImageNet1KRecMetaInfo, imagenet_train_imgrec_iter,\
    imagenet_val_imgrec_iter


def get_dataset_metainfo(dataset_name):
    if dataset_name == "ImageNet1K":
        return ImageNet1KMetaInfo
    elif dataset_name == "ImageNet1K_rec":
        return ImageNet1KRecMetaInfo
    else:
        raise Exception("Unrecognized dataset: {}".format(dataset_name))


def add_dataset_parser_arguments(parser,
                                 dataset_name="ImageNet1K"):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data", dataset_metainfo.root_dir_name),
        help="path to directory with {} dataset".format(dataset_metainfo.label))
    parser.add_argument(
        '--input-size',
        type=int,
        default=dataset_metainfo.input_image_size[0],
        help='size of the input for model')
    parser.add_argument(
        '--resize-inv-factor',
        type=float,
        default=dataset_metainfo.resize_inv_factor,
        help='inverted ratio for input image crop')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=dataset_metainfo.num_classes,
        help='number of classes')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=dataset_metainfo.in_channels,
        help='number of input channels')


def get_batch_fn(dataset_name):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
    if dataset_metainfo.use_imgrec:
        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            return data, label
        return batch_fn
    else:
        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label
        return batch_fn


def get_train_data_source(dataset_name,
                          dataset_dir,
                          batch_size,
                          num_workers,
                          input_image_size=(224, 224)):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
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
        transform_train = imagenet_train_transform(input_image_size=input_image_size)
        dataset = ImageNet1K(
            root=dataset_dir,
            train=True).transform_first(fn=transform_train)
        return gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            last_batch="discard",
            num_workers=num_workers)


def get_val_data_source(dataset_name,
                        dataset_dir,
                        batch_size,
                        num_workers,
                        input_image_size=(224, 224),
                        resize_inv_factor=0.875):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
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
        transform_val = imagenet_val_transform(
            input_image_size=input_image_size,
            resize_value=resize_value)
        dataset = ImageNet1K(
            root=dataset_dir,
            train=False).transform_first(fn=transform_val)
        return gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
