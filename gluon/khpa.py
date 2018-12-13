"""
    KHPA dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_batch_fn', 'get_train_data_source', 'get_val_data_source']

import os
import math
import random
import logging
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--data-path',
        type=str,
        default='../imgclsmob_data/khpa',
        help='path to KHPA dataset')
    parser.add_argument(
        '--split-file',
        type=str,
        default='../imgclsmob_data/khpa/split.csv',
        help='path to file with splitting training subset on training and validation ones')
    parser.add_argument(
        '--generate-split',
        action='store_true',
        help='whether generate split file')
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.1,
        help='fraction of validation subset')


class KHPA(Dataset):
    """
    Load the KHPA classification dataset.

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
                 root=os.path.join('~', '.mxnet', 'datasets', 'khpa'),
                 split_file_path=os.path.join('~', '.mxnet', 'datasets', 'khpa', 'split.csv'),
                 generate_split=False,
                 split_ratio=0.1,
                 num_classes=28,
                 train=True,
                 transform=None):
        super(KHPA, self).__init__()
        self._transform = transform

        root_dir_path = os.path.expanduser(root)
        assert os.path.exists(root_dir_path)

        train_file_name = "train.csv"
        train_file_path = os.path.join(root_dir_path, train_file_name)
        if not os.path.exists(train_file_path):
            raise Exception("Train file doesn't exist: {}".format(train_file_path))

        images_dir_path = os.path.join(root_dir_path, "train")
        if not os.path.exists(images_dir_path):
            raise Exception("Train image directory doesn't exist: {}".format(images_dir_path))

        train_df = pd.read_csv(
            train_file_path,
            sep=',',
            index_col=False,
            dtype={'Id': np.unicode, 'Target': np.unicode})
        train_file_ids = train_df['Id'].values.astype(np.unicode)
        train_file_labels = train_df['Target'].values.astype(np.unicode)

        image_count = len(train_file_ids)

        if os.path.exists(split_file_path):
            if generate_split:
                logging.info('Split file already exists: {}'.format(split_file_path))

            slice_df = pd.read_csv(
                split_file_path,
                sep=',',
                index_col=False,
                dtype={'Id': np.unicode, 'Category': np.int32})
            categories = slice_df['Category'].values
        else:
            if not generate_split:
                raise Exception("Split file doesn't exist: {}".format(split_file_path))

            categories = self.create_slice_category_list(
                count=image_count,
                slice_fraction=split_ratio)

            slice_df = pd.DataFrame({
                'Id': train_file_ids,
                'Category': categories})

            slice_df.to_csv(
                split_file_path,
                sep=',',
                columns=['Id', 'Category'],
                index=False)

        mask = (categories == (1 if train else 2))
        self.train_file_ids = train_file_ids[mask]
        self.train_file_labels = train_file_labels[mask]
        self.images_dir_path = images_dir_path
        self.suffices = ("red", "green", "blue", "yellow")
        self.num_classes = num_classes
        self.train = train

    def __str__(self):
        return self.__class__.__name__ + '({})'.format(len(self.train_file_ids))

    def __len__(self):
        return len(self.train_file_ids)

    def __getitem__(self, idx):
        image_prefix = self.train_file_ids[idx]
        image_prefix_path = os.path.join(self.images_dir_path, image_prefix)

        imgs = []
        for suffix in self.suffices:
            image_file_path = "{}_{}.png".format(image_prefix_path, suffix)
            img = mx.image.imread(image_file_path, flag=0)
            imgs += [img]

        img = mx.nd.concat(*imgs, dim=2)

        if self.train:
            img = self.flip(img)

        labels = self.train_file_labels[idx].split()

        label = np.zeros((self.num_classes, ), np.int32)
        for each_label_str in labels:
            each_label_int = int(each_label_str)
            assert (0 <= each_label_int < self.num_classes)
            label[each_label_int] = 1
        label = mx.nd.array(label)

        # mx.nd.image.normalize(x, self._mean, self._std)

        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    @staticmethod
    def flip(x):
        if bool(random.getrandbits(1)):
            x = mx.nd.flip(x, axis=0)
        if bool(random.getrandbits(1)):
            x = mx.nd.flip(x, axis=1)
        return x

    @staticmethod
    def create_slice_category_list(count, slice_fraction):
        assert (count > 0.0)
        assert (slice_fraction > 0.0)
        index_array = np.arange(count)
        np.random.shuffle(index_array)
        split_at = int(count * slice_fraction)
        sliced2_index_array = index_array[:split_at]
        category_list = np.ones((count,), np.uint8)
        category_list[sliced2_index_array] = 2
        return category_list


def get_batch_fn():
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label
    return batch_fn


def get_train_data_loader(data_dir_path,
                          split_file_path,
                          generate_split,
                          split_ratio,
                          batch_size,
                          num_workers,
                          input_image_size,
                          mean_rgb,
                          std_rgb,
                          jitter_param,
                          lighting_param):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_image_size),
        # transforms.RandomFlipLeftRight(),
        # transforms.RandomColorJitter(
        #     brightness=jitter_param,
        #     contrast=jitter_param,
        #     saturation=jitter_param),
        # transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=mean_rgb,
        #     std=std_rgb)
    ])
    return gluon.data.DataLoader(
        dataset=KHPA(
            root=data_dir_path,
            split_file_path=split_file_path,
            generate_split=generate_split,
            split_ratio=split_ratio,
            train=True).transform_first(fn=transform_train),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)


def get_val_data_loader(data_dir_path,
                        split_file_path,
                        generate_split,
                        split_ratio,
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
        dataset=KHPA(
            root=data_dir_path,
            split_file_path=split_file_path,
            generate_split=generate_split,
            split_ratio=split_ratio,
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

    mean_rgby = (0.485, 0.456, 0.406, 0.406)
    std_rgby = (0.229, 0.224, 0.225, 0.225)

    return get_train_data_loader(
        data_dir_path=dataset_args.data_path,
        split_file_path=dataset_args.split_file,
        generate_split=dataset_args.generate_split,
        split_ratio=dataset_args.split_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        input_image_size=input_image_size,
        mean_rgb=mean_rgby,
        std_rgb=std_rgby,
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

    mean_rgby = (0.485, 0.456, 0.406, 0.406)
    std_rgby = (0.229, 0.224, 0.225, 0.225)

    return get_val_data_loader(
        data_dir_path=dataset_args.data_path,
        split_file_path=dataset_args.split_file,
        generate_split=dataset_args.generate_split,
        split_ratio=dataset_args.split_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        input_image_size=input_image_size,
        resize_value=resize_value,
        mean_rgb=mean_rgby,
        std_rgb=std_rgby)
