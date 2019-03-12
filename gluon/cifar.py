"""
    CIFAR/SVHN dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'get_train_data_source', 'get_val_data_source',
           'get_num_training_samples']

import os
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.utils import download, check_sha1


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
    elif dataset_name == "SVHN":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/svhn',
            help='path to directory with SVHN dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=10,
            help='number of classes')
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')


class CIFAR100Fine(gluon.data.vision.CIFAR100):

    def __init__(self,
                 root,
                 train):
        super(CIFAR100Fine, self).__init__(
            root=root,
            fine_label=True,
            train=train)


class SVHN(gluon.data.dataset._DownloadedDataset):
    """
    SVHN image classification dataset from http://ufldl.stanford.edu/housenumbers/.
    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0`.

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/svhn
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample.
    """
    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets', 'svhn'),
                 train=True,
                 transform=None):
        self._train = train
        self._train_data = [("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "train_32x32.mat",
                             "e6588cae42a1a5ab5efe608cc5cd3fb9aaffd674")]
        self._test_data = [("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "test_32x32.mat",
                            "29b312382ca6b9fba48d41a7b5c19ad9a5462b20")]
        super(SVHN, self).__init__(root, transform)

    def _get_data(self):
        if any(not os.path.exists(path) or not check_sha1(path, sha1) for path, sha1 in
               ((os.path.join(self._root, name), sha1) for _, name, sha1 in self._train_data + self._test_data)):
            for url, _, sha1 in self._train_data + self._test_data:
                download(url=url, path=self._root, sha1_hash=sha1)

        if self._train:
            data_files = self._train_data[0]
        else:
            data_files = self._test_data[0]

        import scipy.io as sio

        loaded_mat = sio.loadmat(os.path.join(self._root, data_files[1]))

        data = loaded_mat['X']
        data = np.transpose(data, (3, 0, 1, 2))
        self._data = mx.nd.array(data, dtype=data.dtype)

        self._label = loaded_mat['y'].astype(np.int32).squeeze()
        np.place(self._label, self._label == 10, 0)


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


def get_num_training_samples(dataset_name):
    if dataset_name == "CIFAR10":
        return 50000
    elif dataset_name == "CIFAR100":
        return 50000
    elif dataset_name == "SVHN":
        return 73257
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))


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
        dataset_class = CIFAR100Fine
    elif dataset_name == "SVHN":
        dataset_class = SVHN
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    dataset = dataset_class(
        root=dataset_dir,
        train=True)

    return gluon.data.DataLoader(
        dataset=dataset.transform_first(fn=transform_train),
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
        dataset_class = CIFAR100Fine
    elif dataset_name == "SVHN":
        dataset_class = SVHN
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    dataset = dataset_class(
        root=dataset_dir,
        train=False)

    return gluon.data.DataLoader(
        dataset=dataset.transform_first(fn=transform_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
