"""
    KHPA dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_batch_fn', 'get_train_data_source', 'get_val_data_source']

import os
import math
import json
import logging
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms
from imgaug import augmenters as iaa
from imgaug import parameters as iap


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
        '--gen-split',
        action='store_true',
        help='whether generate split file')
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.1,
        help='fraction of validation subset')
    parser.add_argument(
        '--stats-file',
        type=str,
        default='../imgclsmob_data/khpa/stats.json',
        help='path to file with the dataset statistics')
    parser.add_argument(
        '--gen-stats',
        action='store_true',
        help='whether generate a file with the dataset statistics')


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
                 stats_file_path=os.path.join('~', '.mxnet', 'datasets', 'khpa', 'stats.json'),
                 generate_stats=False,
                 num_classes=28,
                 model_input_image_size=(224, 224),
                 train=True):
        super(KHPA, self).__init__()
        self.suffices = ("red", "green", "blue", "yellow")

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

        if os.path.exists(stats_file_path):
            if generate_stats:
                logging.info('Stats file already exists: {}'.format(stats_file_path))

            with open(stats_file_path, 'r') as f:
                stats_dict = json.load(f)

            mean_rgby = np.array(stats_dict["mean_rgby"], np.float32)
            std_rgby = np.array(stats_dict["std_rgby"], np.float32)
            label_counts = np.array(stats_dict["label_counts"], np.int32)
        else:
            if not generate_split:
                raise Exception("Stats file doesn't exist: {}".format(stats_file_path))

            label_counts = self.calc_label_counts(train_file_labels, num_classes)
            mean_rgby, std_rgby = self.calc_image_widths(train_file_ids, self.suffices, images_dir_path)
            stats_dict = {
                "mean_rgby": [float(x) for x in mean_rgby],
                "std_rgby": [float(x) for x in std_rgby],
                "label_counts": [int(x) for x in label_counts],
            }
            with open(stats_file_path, 'w') as f:
                json.dump(stats_dict, f)

        self.label_widths = self.calc_label_widths(label_counts, num_classes)

        self.mean_rgby = mean_rgby
        self.std_rgby = std_rgby

        mask = (categories == (1 if train else 2))
        self.train_file_ids = train_file_ids[mask]
        self.train_file_labels = train_file_labels[mask]
        self.images_dir_path = images_dir_path
        self.num_classes = num_classes
        self.train = train

        self._transform = KHPATrainTransform(mean=self.mean_rgby, std=self.std_rgby,
                                             crop_image_size=model_input_image_size) if train else \
            KHPAValTransform(mean=self.mean_rgby, std=self.std_rgby, crop_image_size=model_input_image_size)

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

        label_str_list = self.train_file_labels[idx].split()
        weight = 0.0
        label = np.zeros((self.num_classes, ), np.int32)
        for each_label_str in label_str_list:
            each_label_int = int(each_label_str)
            assert (0 <= each_label_int < self.num_classes)
            label[each_label_int] = 1
            weight += self.label_widths[each_label_int]
        label = mx.nd.array(label)

        if self._transform is not None:
            img, label = self._transform(img, label)
        return img, label, weight

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

    @staticmethod
    def calc_label_counts(train_file_labels, num_classes):
        label_counts = np.zeros((num_classes, ), np.int32)
        for train_file_label in train_file_labels:
            label_str_list = train_file_label.split()
            for label_str in label_str_list:
                label_int = int(label_str)
                assert (0 <= label_int < num_classes)
                label_counts[label_int] += 1
        return label_counts

    @staticmethod
    def calc_label_widths(label_counts, num_classes):
        total_label_count = label_counts.sum()
        label_widths = (1.0 / label_counts) / num_classes * total_label_count
        return label_widths

    @staticmethod
    def calc_image_widths(train_file_ids, suffices, images_dir_path):
        mean_rgby = np.zeros((len(suffices),), np.float32)
        std_rgby = np.zeros((len(suffices),), np.float32)
        for i, suffix in enumerate(suffices):
            imgs = []
            for image_prefix in train_file_ids:
                image_prefix_path = os.path.join(images_dir_path, image_prefix)
                image_file_path = "{}_{}.png".format(image_prefix_path, suffix)
                img = mx.image.imread(image_file_path, flag=0).asnumpy()
                imgs += [img]
                if len(imgs) > 10:
                    break
            imgs = np.concatenate(tuple(imgs), axis=2)
            mean_rgby[i] = imgs.mean()
            std_rgby[i] = imgs.std()
        return mean_rgby, std_rgby


class KHPATrainTransform(object):
    def __init__(self,
                 mean=(0.0, 0.0, 0.0, 0.0),
                 std=(1.0, 1.0, 1.0, 1.0),
                 crop_image_size=(224, 224)):
        if isinstance(crop_image_size, int):
            crop_image_size = (crop_image_size, crop_image_size)
        self._mean = mean
        self._std = std
        self.crop_image_size = crop_image_size

        self.seq = iaa.Sequential(
            children=[
                iaa.Sequential(
                    children=[
                        iaa.Fliplr(
                            p=0.5,
                            name="Fliplr"),
                        iaa.Flipud(
                            p=0.5,
                            name="Flipud"),
                        iaa.Sequential(
                            children=[
                                iaa.Affine(
                                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                    rotate=(-45, 45),
                                    shear=(-16, 16),
                                    order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                    mode="reflect",
                                    name="Affine"),
                                iaa.Sometimes(
                                    p=0.01,
                                    then_list=iaa.PiecewiseAffine(
                                        scale=(0.0, 0.01),
                                        nb_rows=(4, 20),
                                        nb_cols=(4, 20),
                                        order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                        mode="reflect",
                                        name="PiecewiseAffine"))],
                            random_order=True,
                            name="GeomTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.75,
                                    then_list=iaa.Add(
                                        value=(-10, 10),
                                        per_channel=0.5,
                                        name="Brightness")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.Emboss(
                                        alpha=(0.0, 0.5),
                                        strength=(0.5, 1.2),
                                        name="Emboss")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.Sharpen(
                                        alpha=(0.0, 0.5),
                                        lightness=(0.5, 1.2),
                                        name="Sharpen")),
                                iaa.Sometimes(
                                    p=0.25,
                                    then_list=iaa.ContrastNormalization(
                                        alpha=(0.5, 1.5),
                                        per_channel=0.5,
                                        name="ContrastNormalization"))
                            ],
                            random_order=True,
                            name="ColorTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.AdditiveGaussianNoise(
                                        loc=0,
                                        scale=(0.0, 10.0),
                                        per_channel=0.5,
                                        name="AdditiveGaussianNoise")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.SaltAndPepper(
                                        p=(0, 0.001),
                                        per_channel=0.5,
                                        name="SaltAndPepper"))],
                            random_order=True,
                            name="Noise"),
                        iaa.OneOf(
                            children=[
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.MedianBlur(
                                        k=3,
                                        name="MedianBlur")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.AverageBlur(
                                        k=(2, 4),
                                        name="AverageBlur")),
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.GaussianBlur(
                                        sigma=(0.0, 2.0),
                                        name="GaussianBlur"))],
                            name="Blur"),
                    ],
                    random_order=True,
                    name="MainProcess")])

    def __call__(self, img, label):

        seq_det = self.seq.to_deterministic()
        imgs_aug = seq_det.augment_images(img.asnumpy().transpose((2, 0, 1)))
        img_np = imgs_aug.transpose((1, 2, 0))
        img_np = img_np.astype(np.float32) / 255.0
        img_np = (img_np - self._mean) / self._std
        img = mx.nd.array(img_np, ctx=img.context)
        img = mx.image.random_size_crop(
            img,
            size=self.crop_image_size,
            area=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interp=1)[0]
        return img, label


class KHPAValTransform(object):
    def __init__(self,
                 mean=(0.0, 0.0, 0.0, 0.0),
                 std=(1.0, 1.0, 1.0, 1.0),
                 crop_image_size=(224, 224)):
        if isinstance(crop_image_size, int):
            crop_image_size = (crop_image_size, crop_image_size)
        self._mean = mean
        self._std = std
        self.crop_image_size = crop_image_size

    def __call__(self, img, label):
        img = mx.nd.image.to_tensor(img)
        img = (img - mx.nd.array(self._mean, ctx=img.context)) / mx.nd.array(self._std, ctx=img.context)
        return img, label


def get_batch_fn():
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        weight = gluon.utils.split_and_load(batch[2].astype(np.float32, copy=False), ctx_list=ctx, batch_axis=0)
        return data, label, weight
    return batch_fn


def get_train_data_loader(data_dir_path,
                          split_file_path,
                          generate_split,
                          split_ratio,
                          stats_file_path,
                          generate_stats,
                          batch_size,
                          num_workers,
                          model_input_image_size):
    return gluon.data.DataLoader(
        dataset=KHPA(
            root=data_dir_path,
            split_file_path=split_file_path,
            generate_split=generate_split,
            split_ratio=split_ratio,
            stats_file_path=stats_file_path,
            generate_stats=generate_stats,
            model_input_image_size=model_input_image_size,
            train=True),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)


def get_val_data_loader(data_dir_path,
                        split_file_path,
                        generate_split,
                        split_ratio,
                        stats_file_path,
                        generate_stats,
                        batch_size,
                        num_workers,
                        model_input_image_size,
                        resize_value):
    transform_test = transforms.Compose([
        transforms.Resize(resize_value, keep_ratio=True),
        transforms.CenterCrop(model_input_image_size),
    ])
    return gluon.data.DataLoader(
        dataset=KHPA(
            root=data_dir_path,
            split_file_path=split_file_path,
            generate_split=generate_split,
            split_ratio=split_ratio,
            stats_file_path=stats_file_path,
            generate_stats=generate_stats,
            model_input_image_size=model_input_image_size,
            train=False).transform_first(fn=transform_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)


def get_train_data_source(dataset_args,
                          batch_size,
                          num_workers,
                          input_image_size=(224, 224)):
    return get_train_data_loader(
        data_dir_path=dataset_args.data_path,
        split_file_path=dataset_args.split_file,
        generate_split=dataset_args.gen_split,
        split_ratio=dataset_args.split_ratio,
        stats_file_path=dataset_args.stats_file,
        generate_stats=dataset_args.gen_stats,
        batch_size=batch_size,
        num_workers=num_workers,
        model_input_image_size=input_image_size)


def get_val_data_source(dataset_args,
                        batch_size,
                        num_workers,
                        input_image_size=(224, 224),
                        resize_inv_factor=0.875):
    assert (resize_inv_factor > 0.0)
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))

    return get_val_data_loader(
        data_dir_path=dataset_args.data_path,
        split_file_path=dataset_args.split_file,
        generate_split=dataset_args.gen_split,
        split_ratio=dataset_args.split_ratio,
        stats_file_path=dataset_args.stats_file,
        generate_stats=dataset_args.gen_stats,
        batch_size=batch_size,
        num_workers=num_workers,
        model_input_image_size=input_image_size,
        resize_value=resize_value)
