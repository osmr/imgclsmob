"""
    CUB-200-2011 fine-grained classification dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'get_train_data_source', 'get_val_data_source']

import math
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from .cub200_2011_cls_dataset import CUB200_2011


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


def get_train_data_source(dataset_dir,
                          batch_size,
                          num_workers,
                          input_image_size=(224, 224)):
    jitter_param = 0.4
    lighting_param = 0.1

    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

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

    dataset = CUB200_2011(
        root=dataset_dir,
        train=True).transform_first(fn=transform_train)

    # num_training_samples = len(dataset)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)


def get_val_data_source(dataset_dir,
                        batch_size,
                        num_workers,
                        input_image_size=(224, 224),
                        resize_inv_factor=0.875):
    assert (resize_inv_factor > 0.0)
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))

    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform_val = transforms.Compose([
        transforms.Resize(resize_value, keep_ratio=True),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
    return gluon.data.DataLoader(
        dataset=CUB200_2011(
            root=dataset_dir,
            train=False).transform_first(fn=transform_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
