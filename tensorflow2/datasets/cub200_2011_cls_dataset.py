"""
    CUB-200-2011 classification dataset.
"""

import os
import numpy as np
import pandas as pd
import threading
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from .cls_dataset import img_normalization
from .imagenet1k_cls_dataset import ImageNet1KMetaInfo


class CUBDirectoryIterator(DirectoryIterator):
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 mode="val"):
        super(CUBDirectoryIterator, self).set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation)

        root_dir_path = os.path.expanduser(directory)
        assert os.path.exists(root_dir_path)

        images_file_name = "images.txt"
        images_file_path = os.path.join(root_dir_path, images_file_name)
        if not os.path.exists(images_file_path):
            raise Exception("Images file doesn't exist: {}".format(images_file_name))

        class_file_name = "image_class_labels.txt"
        class_file_path = os.path.join(root_dir_path, class_file_name)
        if not os.path.exists(class_file_path):
            raise Exception("Image class file doesn't exist: {}".format(class_file_name))

        split_file_name = "train_test_split.txt"
        split_file_path = os.path.join(root_dir_path, split_file_name)
        if not os.path.exists(split_file_path):
            raise Exception("Split file doesn't exist: {}".format(split_file_name))

        images_df = pd.read_csv(
            images_file_path,
            sep="\s+",
            header=None,
            index_col=False,
            names=["image_id", "image_path"],
            dtype={"image_id": np.int32, "image_path": np.unicode})
        class_df = pd.read_csv(
            class_file_path,
            sep="\s+",
            header=None,
            index_col=False,
            names=["image_id", "class_id"],
            dtype={"image_id": np.int32, "class_id": np.uint8})
        split_df = pd.read_csv(
            split_file_path,
            sep="\s+",
            header=None,
            index_col=False,
            names=["image_id", "split_flag"],
            dtype={"image_id": np.int32, "split_flag": np.uint8})
        df = images_df.join(class_df, rsuffix="_class_df").join(split_df, rsuffix="_split_df")
        split_flag = 1 if mode == "train" else 0
        subset_df = df[df.split_flag == split_flag]

        image_ids = subset_df["image_id"].values.astype(np.int32)
        class_ids = subset_df["class_id"].values.astype(np.int32) - 1
        image_file_names = subset_df["image_path"].values.astype(np.unicode)

        images_dir_name = "images"
        self.images_dir_path = os.path.join(root_dir_path, images_dir_name)
        assert os.path.exists(self.images_dir_path)
        assert (len(image_ids) == len(class_ids))

        self.class_mode = class_mode
        self.dtype = dtype
        self._filepaths = [os.path.join(self.images_dir_path, image_file_name) for image_file_name in image_file_names]
        self.classes = [int(class_id) for class_id in class_ids]

        self.n = len(class_ids)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()


class CubImageDataGenerator(ImageDataGenerator):

    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest',
                            mode="val"):
        return CUBDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            mode=mode)


class CUB200MetaInfo(ImageNet1KMetaInfo):
    def __init__(self):
        super(CUB200MetaInfo, self).__init__()
        self.label = "CUB200_2011"
        self.short_label = "cub"
        self.root_dir_name = "CUB_200_2011"
        self.dataset_class = None
        self.num_training_samples = None
        self.num_classes = 200
        self.train_metric_capts = ["Train.Err"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err"}]
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.saver_acc_ind = 0
        self.train_transform = cub200_train_transform
        self.val_transform = cub200_val_transform
        self.test_transform = cub200_val_transform
        self.train_generator = cub200_train_generator
        self.val_generator = cub200_val_generator
        self.test_generator = cub200_val_generator
        self.net_extra_kwargs = {"aux": False}
        self.load_ignore_extra = True

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        super(CUB200MetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--no-aux",
            dest="no_aux",
            action="store_true",
            help="no `aux` mode in model")

    def update(self,
               args):
        super(CUB200MetaInfo, self).update(args)
        if args.no_aux:
            self.net_extra_kwargs = None
            self.load_ignore_extra = False


def cub200_train_transform(ds_metainfo,
                           data_format="channels_last"):
    """
    Create image transform sequence for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        CUB-200-2011 dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = CubImageDataGenerator(
        preprocessing_function=(lambda img: img_normalization(
            img=img,
            mean_rgb=ds_metainfo.mean_rgb,
            std_rgb=ds_metainfo.std_rgb)),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format=data_format)
    return data_generator


def cub200_val_transform(ds_metainfo,
                         data_format="channels_last"):
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        CUB-200-2011 dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = CubImageDataGenerator(
        preprocessing_function=(lambda img: img_normalization(
            img=img,
            mean_rgb=ds_metainfo.mean_rgb,
            std_rgb=ds_metainfo.std_rgb)),
        data_format=data_format)
    return data_generator


def cub200_train_generator(data_generator,
                           ds_metainfo,
                           batch_size):
    """
    Create image generator for training subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    root = ds_metainfo.root_dir_path
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation=ds_metainfo.interpolation_msg,
        mode="val")
    return generator


def cub200_val_generator(data_generator,
                         ds_metainfo,
                         batch_size):
    """
    Create image generator for validation subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    root = ds_metainfo.root_dir_path
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation=ds_metainfo.interpolation_msg,
        mode="val")
    return generator
