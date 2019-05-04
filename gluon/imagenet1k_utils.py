"""
    ImageNet-1K dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_batch_fn', 'get_train_data_source', 'get_val_data_source',
           'num_training_samples']

from mxnet import gluon
from gluon.datasets.imagenet1k_cls_dataset import ImageNet, calc_val_resize_value, imagenet_val_transform,\
    imagenet_train_transform
from gluon.datasets.imagenet1k_rec_cls_dataset import imagenet_train_imgrec_iter, imagenet_val_imgrec_iter


num_training_samples = 1281167


def add_dataset_parser_arguments(parser,
                                 dataset_name="ImageNet1K"):
    if dataset_name != "ImageNet1K":
        raise ValueError('Unrecognized dataset: {}'.format(dataset_name))
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/imagenet',
        help='path to directory with ImageNet-1K dataset')
    parser.add_argument(
        '--rec-train',
        type=str,
        default='../imgclsmob_data/imagenet/rec/train.rec',
        help='the ImageNet-1K training data')
    parser.add_argument(
        '--rec-train-idx',
        type=str,
        default='../imgclsmob_data/imagenet/rec/train.idx',
        help='the index of ImageNet-1K training data')
    parser.add_argument(
        '--rec-val',
        type=str,
        default='../imgclsmob_data/imagenet/rec/val.rec',
        help='the ImageNet-1K validation data')
    parser.add_argument(
        '--rec-val-idx',
        type=str,
        default='../imgclsmob_data/imagenet/rec/val.idx',
        help='the index of ImageNet-1K validation data')
    parser.add_argument(
        '--use-rec',
        action='store_true',
        help='use image record iter for ImageNet-1K data input')

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
        default=1000,
        help='number of classes')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')


def get_batch_fn(dataset_args):
    if dataset_args.use_rec:
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


def get_train_data_source(dataset_args,
                          batch_size,
                          num_workers,
                          input_image_size=(224, 224)):
    if dataset_args.use_rec:
        if isinstance(input_image_size, int):
            input_image_size = (input_image_size, input_image_size)
        data_shape = (3,) + input_image_size
        return imagenet_train_imgrec_iter(
            imgrec_file_path=dataset_args.rec_train,
            imgidx_file_path=dataset_args.rec_train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            data_shape=data_shape)
    else:
        transform_train = imagenet_train_transform(input_image_size=input_image_size)
        dataset = ImageNet(
            root=dataset_args.data_dir,
            train=True).transform_first(fn=transform_train)
        return gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            last_batch="discard",
            num_workers=num_workers)


def get_val_data_source(dataset_args,
                        batch_size,
                        num_workers,
                        input_image_size=(224, 224),
                        resize_inv_factor=0.875):
    resize_value = calc_val_resize_value(
        input_image_size=input_image_size,
        resize_inv_factor=resize_inv_factor)
    if dataset_args.use_rec:
        if isinstance(input_image_size, int):
            input_image_size = (input_image_size, input_image_size)
        data_shape = (3,) + input_image_size
        return imagenet_val_imgrec_iter(
            imgrec_file_path=dataset_args.rec_val,
            imgidx_file_path=dataset_args.rec_val_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            data_shape=data_shape,
            resize_value=resize_value)
    else:
        transform_val = imagenet_val_transform(
            input_image_size=input_image_size,
            resize_value=resize_value)
        dataset = ImageNet(
            root=dataset_args.data_dir,
            train=False).transform_first(fn=transform_val)
        return gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
