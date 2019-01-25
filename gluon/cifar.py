"""
    CIFAR dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'get_train_data_source', 'get_val_data_source',
           'num_training_samples']

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block
from mxnet.gluon.data.vision import transforms


num_training_samples = 50000


def add_dataset_parser_arguments(parser,
                                 dataset_name):
    if dataset_name == "CIFAR10":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/cifar10',
            help='path to directory with CIFAR-10 dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=10,
            help='number of classes')
    elif dataset_name == "CIFAR100":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/cifar100',
            help='path to directory with CIFAR-100 dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=100,
            help='number of classes')
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')


class RandomCrop(Block):
    """Randomly crop `src` with `size` (width, height).
    Padding is optional.
    Upsample result if `src` is smaller than `size`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of the final output.
    pad: int or tuple
        if int, size of the zero-padding
        if tuple, number of values padded to the edges of each axis.
            ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
            ((before, after),) yields same before and after pad for each axis.
            (pad,) or int is a shortcut for before = after = pad width for all axes.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
    Outputs:
        - **out**: output tensor with ((H+2*pad) x (W+2*pad) x C) shape.
    """

    def __init__(self, size, pad=None, interpolation=2):
        super(RandomCrop, self).__init__()
        numeric_types = (float, int, np.generic)
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, interpolation)
        if isinstance(pad, int):
            self.pad = ((pad, pad), (pad, pad), (0, 0))
        else:
            self.pad = pad

    def forward(self, x):
        if self.pad:
            x_pad = np.pad(x.asnumpy(), self.pad,
                           mode='constant', constant_values=0)

        return mx.image.random_crop(mx.nd.array(x_pad), *self._args)[0]


def batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label


def get_train_data_source(dataset_name,
                          dataset_dir,
                          batch_size,
                          num_workers):
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = (0.4914, 0.4822, 0.4465)
    std_rgb = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        RandomCrop(size=32, pad=4),
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

    if dataset_name == "CIFAR10":
        dataset_class = gluon.data.vision.CIFAR10
    elif dataset_name == "CIFAR100":
        dataset_class = gluon.data.vision.CIFAR100
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    return gluon.data.DataLoader(
        dataset=dataset_class(
            root=dataset_dir,
            train=True).transform_first(fn=transform_train),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)


def get_val_data_source(dataset_name,
                        dataset_dir,
                        batch_size,
                        num_workers):
    mean_rgb = (0.4914, 0.4822, 0.4465)
    std_rgb = (0.2023, 0.1994, 0.2010)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])

    if dataset_name == "CIFAR10":
        dataset_class = gluon.data.vision.CIFAR10
    elif dataset_name == "CIFAR100":
        dataset_class = gluon.data.vision.CIFAR100
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    return gluon.data.DataLoader(
        dataset=dataset_class(
            root=dataset_dir,
            train=False).transform_first(fn=transform_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
