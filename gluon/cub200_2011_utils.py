"""
    CUB-200-2011 fine-grained classification dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'cub200_train_data_loader', 'cub200_val_data_loader']

from mxnet import gluon
from gluon.datasets.cub200_2011_cls_dataset import CUB200_2011
from gluon.datasets.imagenet1k_cls_dataset import imagenet_train_transform, imagenet_val_transform,\
    calc_val_resize_value


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/CUB_200_2011',
        help='path to directory with CUB-200-2011 dataset')
    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='size of the input for model')
    parser.add_argument(
        '--resize-inv-factor',
        type=float,
        default=0.875,
        help='inverted ratio for input image crop')

    parser.add_argument(
        '--num-classes',
        type=int,
        default=200,
        help='number of classes')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')


def batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label


def cub200_train_data_loader(dataset_dir,
                             batch_size,
                             num_workers,
                             input_image_size=(224, 224)):
    transform_train = imagenet_train_transform(input_image_size=input_image_size)
    dataset = CUB200_2011(
        root=dataset_dir,
        train=True).transform_first(fn=transform_train)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        last_batch="discard",
        num_workers=num_workers)


def cub200_val_data_loader(dataset_dir,
                           batch_size,
                           num_workers,
                           input_image_size=(224, 224),
                           resize_inv_factor=0.875):
    resize_value = calc_val_resize_value(
        input_image_size=input_image_size,
        resize_inv_factor=resize_inv_factor)
    transform_val = imagenet_val_transform(
        input_image_size=input_image_size,
        resize_value=resize_value)
    dataset = CUB200_2011(
        root=dataset_dir,
        train=False).transform_first(fn=transform_val)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
