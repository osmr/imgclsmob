"""
    CIFAR/SVHN dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'get_train_data_source', 'get_val_data_source',
           'get_num_training_samples']

import os
import tarfile
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url


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


class SVHN(gluon.dataset._DownloadedDataset):
    """
    CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html
    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar10
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample.
    """
    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets', 'cifar10'),
                 train=True,
                 transform=None):
        self._train = train
        self._archive_file = ('cifar-10-binary.tar.gz', 'fab780a1e191a7eda0f345501ccd62d20f7ed891')
        self._train_data = [('data_batch_1.bin', 'aadd24acce27caa71bf4b10992e9e7b2d74c2540'),
                            ('data_batch_2.bin', 'c0ba65cce70568cd57b4e03e9ac8d2a5367c1795'),
                            ('data_batch_3.bin', '1dd00a74ab1d17a6e7d73e185b69dbf31242f295'),
                            ('data_batch_4.bin', 'aab85764eb3584312d3c7f65fd2fd016e36a258e'),
                            ('data_batch_5.bin', '26e2849e66a845b7f1e4614ae70f4889ae604628')]
        self._test_data = [('test_batch.bin', '67eb016db431130d61cd03c7ad570b013799c88c')]
        self._namespace = 'cifar10'
        super(SVHN, self).__init__(root, transform)

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072 + 1)

        return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), data[:, 0].astype(np.int32)

    def _get_data(self):
        if any(not os.path.exists(path) or not check_sha1(path, sha1)
               for path, sha1 in ((os.path.join(self._root, name), sha1)
                                  for name, sha1 in self._train_data + self._test_data)):
            namespace = 'gluon/dataset/' + self._namespace
            filename = download(_get_repo_file_url(namespace, self._archive_file[0]),
                                path=self._root,
                                sha1_hash=self._archive_file[1])

            with tarfile.open(filename) as tar:
                tar.extractall(self._root)

        if self._train:
            data_files = self._train_data
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name, _ in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)

        self._data = mx.nd.array(data, dtype=data.dtype)
        self._label = label


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
        # dataset_class = gluon.data.vision.CIFAR10
        dataset = gluon.data.vision.CIFAR10(
            root=dataset_dir,
            train=True)
    elif dataset_name == "CIFAR100":
        # dataset_class = CIFAR100Fine
        dataset = CIFAR100Fine(
            root=dataset_dir,
            train=True)
    elif dataset_name == "SVHN":
        dataset = SVHN(
            root=dataset_dir,
            train=True)
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

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
        # dataset_class = gluon.data.vision.CIFAR10
        dataset = gluon.data.vision.CIFAR10(
            root=dataset_dir,
            train=False)
    elif dataset_name == "CIFAR100":
        # dataset_class = CIFAR100Fine
        dataset = CIFAR100Fine(
            root=dataset_dir,
            train=False)
    elif dataset_name == "SVHN":
        dataset = SVHN(
            root=dataset_dir,
            train=False)
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    return gluon.data.DataLoader(
        dataset=dataset.transform_first(fn=transform_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
