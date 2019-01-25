"""
    CIFAR dataset routines.
"""

import numpy as np
import chainer
from chainer import iterators
from chainer import Chain
from chainer.dataset import DatasetMixin
from chainer.datasets import cifar

__all__ = ['add_dataset_parser_arguments', 'get_val_data_iterator', 'get_data_iterators', 'CIFARPredictor']


def add_dataset_parser_arguments(parser,
                                 dataset_name):
    if dataset_name == "CIFAR10":
        parser.add_argument(
            '--num-classes',
            type=int,
            default=10,
            help='number of classes')
    elif dataset_name == "CIFAR100":
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


class CIFARPredictor(Chain):

    def __init__(self,
                 base_model,
                 mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        super(CIFARPredictor, self).__init__()
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]
        with self.init_scope():
            self.model = base_model

    def _preprocess(self, img):
        img -= self.mean
        img /= self.std
        return img

    def predict(self, imgs):
        imgs = self.xp.asarray([self._preprocess(img) for img in imgs])

        with chainer.using_config('train', False), chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            predictions = self.model(imgs)

        output = chainer.backends.cuda.to_cpu(predictions.array)
        return output


def get_val_data_iterator(dataset_name,
                          batch_size,
                          num_workers):

    if dataset_name == "CIFAR10":
        _, test_ds = cifar.get_cifar10()
    elif dataset_name == "CIFAR100":
        _, test_ds = cifar.get_cifar100()
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    val_dataset = test_ds
    val_dataset_len = len(val_dataset)

    val_iterator = iterators.MultiprocessIterator(
        dataset=val_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=False,
        n_processes=num_workers,
        shared_mem=300000000)

    return val_iterator, val_dataset_len


class PreprocessedCIFARDataset(DatasetMixin):

    def __init__(self,
                 train,
                 mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        train_ds, test_ds = cifar.get_cifar10()
        self.base = train_ds if train else test_ds
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]

    def __len__(self):
        return len(self.base)

    def _preprocess(self, img):
        img -= self.mean
        img /= self.std
        return img

    def get_example(self, i):
        image, label = self.base[i]
        image = self._preprocess(image)
        return image, label


def get_data_iterators(batch_size,
                       num_workers):

    train_dataset = PreprocessedCIFARDataset(train=True)
    train_iterator = iterators.MultiprocessIterator(
        dataset=train_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=True,
        n_processes=num_workers)

    val_dataset = PreprocessedCIFARDataset(train=False)
    val_iterator = iterators.MultiprocessIterator(
        dataset=val_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=False,
        n_processes=num_workers)

    return train_iterator, val_iterator
