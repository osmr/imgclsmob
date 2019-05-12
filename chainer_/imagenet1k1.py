import math
import os
import numpy as np

import chainer
from chainer import iterators
from chainer import Chain
from chainer.dataset import DatasetMixin

from chainercv.transforms import scale
from chainercv.transforms import center_crop
from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset

__all__ = ['add_dataset_parser_arguments', 'get_val_data_iterator', 'get_data_iterators', 'ImagenetPredictor']


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/imagenet',
        help='path to directory with ImageNet-1K dataset')

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


class ImagenetPredictor(Chain):

    def __init__(self,
                 base_model,
                 scale_size=256,
                 crop_size=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super(ImagenetPredictor, self).__init__()
        self.scale_size = scale_size
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]
        with self.init_scope():
            self.model = base_model

    def _preprocess(self, img):
        img = scale(img=img, size=self.scale_size)
        img = center_crop(img, self.crop_size)
        img /= 255.0
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


class PreprocessedDataset(DatasetMixin):

    def __init__(self,
                 root,
                 scale_size=256,
                 crop_size=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.base = DirectoryParsingLabelDataset(root)
        self.scale_size = scale_size
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]

    def __len__(self):
        return len(self.base)

    def _preprocess(self, img):
        img = scale(img=img, size=self.scale_size)
        img = center_crop(img, self.crop_size)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img

    def get_example(self, i):
        image, label = self.base[i]
        image = self._preprocess(image)
        return image, label


def get_val_data_iterator(data_dir,
                          batch_size,
                          num_workers,
                          num_classes):

    val_dir_path = os.path.join(data_dir, 'val')
    val_dataset = DirectoryParsingLabelDataset(val_dir_path)
    val_dataset_len = len(val_dataset)
    assert(len(directory_parsing_label_names(val_dir_path)) == num_classes)

    val_iterator = iterators.MultiprocessIterator(
        dataset=val_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=False,
        n_processes=num_workers,
        shared_mem=300000000)

    return val_iterator, val_dataset_len


def get_data_iterators(data_dir,
                       batch_size,
                       num_workers,
                       num_classes,
                       input_image_size=224,
                       resize_inv_factor=0.875):
    assert (resize_inv_factor > 0.0)
    resize_value = int(math.ceil(float(input_image_size) / resize_inv_factor))

    train_dir_path = os.path.join(data_dir, 'train')
    train_dataset = PreprocessedDataset(
        root=train_dir_path,
        scale_size=resize_value,
        crop_size=input_image_size)
    assert(len(directory_parsing_label_names(train_dir_path)) == num_classes)

    val_dir_path = os.path.join(data_dir, 'val')
    val_dataset = PreprocessedDataset(
        root=val_dir_path,
        scale_size=resize_value,
        crop_size=input_image_size)
    assert (len(directory_parsing_label_names(val_dir_path)) == num_classes)

    train_iterator = iterators.MultiprocessIterator(
        dataset=train_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=True,
        n_processes=num_workers)

    val_iterator = iterators.MultiprocessIterator(
        dataset=val_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=False,
        n_processes=num_workers)

    return train_iterator, val_iterator
