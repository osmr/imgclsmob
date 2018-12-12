"""
    ImageNet-1K dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_batch_fn', 'get_train_data_source', 'get_val_data_source',
           'num_training_samples']

import os
import math
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.vision import ImageFolderDataset


num_training_samples = 1281167


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--in1k-data-dir',
        type=str,
        default='../imgclsmob_data/imagenet',
        help='path to directory with ImageNet-1K dataset')
    parser.add_argument(
        '--in1k-rec-train',
        type=str,
        default='../imgclsmob_data/imagenet/rec/train.rec',
        help='the ImageNet-1K training data')
    parser.add_argument(
        '--in1k-rec-train-idx',
        type=str,
        default='../imgclsmob_data/imagenet/rec/train.idx',
        help='the index of ImageNet-1K training data')
    parser.add_argument(
        '--in1k-rec-val',
        type=str,
        default='../imgclsmob_data/imagenet/rec/val.rec',
        help='the ImageNet-1K validation data')
    parser.add_argument(
        '--in1k-rec-val-idx',
        type=str,
        default='../imgclsmob_data/imagenet/rec/val.idx',
        help='the index of ImageNet-1K validation data')
    parser.add_argument(
        '--in1k-use-rec',
        action='store_true',
        help='use image record iter for ImageNet-1K data input')


class ImageNet(ImageFolderDataset):
    """
    Load the ImageNet classification dataset.

    Refer to :doc:`../build/examples_datasets/imagenet` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets', 'imagenet'),
                 train=True,
                 transform=None):
        split = 'train' if train else 'val'
        root = os.path.join(root, split)
        super(ImageNet, self).__init__(root=root, flag=1, transform=transform)


def get_batch_fn(dataset_args):
    if dataset_args.in1k_use_rec:
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


def get_train_data_rec(rec_train,
                       rec_train_idx,
                       batch_size,
                       num_workers,
                       data_shape,
                       mean_rgb,
                       std_rgb,
                       jitter_param,
                       lighting_param):
    assert isinstance(data_shape, tuple) and len(data_shape) == 3
    return mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=num_workers,
        shuffle=True,
        batch_size=batch_size,

        data_shape=data_shape,
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=(4. / 3.),
        min_aspect_ratio=(3. / 4.),
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
    )


def get_val_data_rec(rec_val,
                     rec_val_idx,
                     batch_size,
                     num_workers,
                     data_shape,
                     resize_value,
                     mean_rgb,
                     std_rgb):
    assert isinstance(data_shape, tuple) and len(data_shape) == 3
    return mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,

        resize=resize_value,
        data_shape=data_shape,
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )


def get_train_data_loader(data_dir,
                          batch_size,
                          num_workers,
                          input_image_size,
                          mean_rgb,
                          std_rgb,
                          jitter_param,
                          lighting_param):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_image_size),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
    return gluon.data.DataLoader(
        dataset=ImageNet(
            root=data_dir,
            train=True).transform_first(fn=transform_train),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)


def get_val_data_loader(data_dir,
                        batch_size,
                        num_workers,
                        input_image_size,
                        resize_value,
                        mean_rgb,
                        std_rgb):
    transform_test = transforms.Compose([
        transforms.Resize(resize_value, keep_ratio=True),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
    return gluon.data.DataLoader(
        dataset=ImageNet(
            root=data_dir,
            train=False).transform_first(fn=transform_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)


def get_train_data_source(dataset_args,
                          batch_size,
                          num_workers,
                          input_image_size=(224, 224)):
    jitter_param = 0.4
    lighting_param = 0.1

    if dataset_args.in1k_use_rec:
        if isinstance(input_image_size, int):
            input_image_size = (input_image_size, input_image_size)
        data_shape = (3,) + input_image_size
        mean_rgb = (123.68, 116.779, 103.939)
        std_rgb = (58.393, 57.12, 57.375)

        return get_train_data_rec(
            rec_train=dataset_args.in1k_rec_train,
            rec_train_idx=dataset_args.in1k_rec_train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            data_shape=data_shape,
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            jitter_param=jitter_param,
            lighting_param=lighting_param)
    else:
        mean_rgb = (0.485, 0.456, 0.406)
        std_rgb = (0.229, 0.224, 0.225)

        return get_train_data_loader(
            data_dir=dataset_args.in1k_data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            input_image_size=input_image_size,
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            jitter_param=jitter_param,
            lighting_param=lighting_param)


def get_val_data_source(dataset_args,
                        batch_size,
                        num_workers,
                        input_image_size=(224, 224),
                        resize_inv_factor=0.875):
    assert (resize_inv_factor > 0.0)
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))

    if dataset_args.in1k_use_rec:
        data_shape = (3,) + input_image_size
        mean_rgb = (123.68, 116.779, 103.939)
        std_rgb = (58.393, 57.12, 57.375)

        return get_val_data_rec(
            rec_val=dataset_args.in1k_rec_val,
            rec_val_idx=dataset_args.in1k_rec_val_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            data_shape=data_shape,
            resize_value=resize_value,
            mean_rgb=mean_rgb,
            std_rgb=std_rgb)
    else:
        mean_rgb = (0.485, 0.456, 0.406)
        std_rgb = (0.229, 0.224, 0.225)

        return get_val_data_loader(
            data_dir=dataset_args.in1k_data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            input_image_size=input_image_size,
            resize_value=resize_value,
            mean_rgb=mean_rgb,
            std_rgb=std_rgb)
