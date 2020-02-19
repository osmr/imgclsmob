"""
    Pascal VOC2012 semantic segmentation dataset.
"""

import os
import numpy as np
from PIL import Image
from chainer import get_dtype
from .seg_dataset import SegDataset, SegImageDataGenerator
from .dataset_metainfo import DatasetMetaInfo


class VOCSegDataset(SegDataset):
    """
    Pascal VOC2012 semantic segmentation dataset.

    Parameters
    ----------
    root : str
        Path to VOCdevkit folder.
    mode : str, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 **kwargs):
        super(VOCSegDataset, self).__init__(
            root=root,
            mode=mode,
            transform=transform,
            **kwargs)

        base_dir_path = os.path.join(root, "VOC2012")
        image_dir_path = os.path.join(base_dir_path, "JPEGImages")
        mask_dir_path = os.path.join(base_dir_path, "SegmentationClass")

        splits_dir_path = os.path.join(base_dir_path, "ImageSets", "Segmentation")
        if mode == "train":
            split_file_path = os.path.join(splits_dir_path, "train.txt")
        elif mode in ("val", "test", "demo"):
            split_file_path = os.path.join(splits_dir_path, "val.txt")
        else:
            raise RuntimeError("Unknown dataset splitting mode")

        self.images = []
        self.masks = []
        with open(os.path.join(split_file_path), "r") as lines:
            for line in lines:
                image_file_path = os.path.join(image_dir_path, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(image_file_path)
                self.images.append(image_file_path)
                mask_file_path = os.path.join(mask_dir_path, line.rstrip('\n') + ".png")
                assert os.path.isfile(mask_file_path)
                self.masks.append(mask_file_path)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.mode == "demo":
            image = self._img_transform(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        elif self.mode == "val":
            image, mask = self._val_sync_transform(image, mask)
        else:
            assert self.mode == "test"
            image, mask = self._img_transform(image), self._mask_transform(mask)

        if self.transform is not None:
            image = self.transform(image)

        # print("---> image.shape={}".format(image.shape))
        # print("---> mask.shape={}".format(mask.shape))
        return image, mask

    classes = 21
    vague_idx = 255
    use_vague = True
    background_idx = 0
    ignore_bg = True

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        # np_mask[np_mask == 255] = VOCSegDataset.vague_idx
        return np_mask

    def __len__(self):
        return len(self.images)


class VOCSegTrainTransform(object):
    """
    ImageNet-1K training transform.
    """
    def __init__(self,
                 ds_metainfo,
                 mean_rgb=(0.485, 0.456, 0.406),
                 std_rgb=(0.229, 0.224, 0.225)):
        assert (ds_metainfo is not None)
        self.mean = np.array(mean_rgb, np.float32)[np.newaxis, np.newaxis, :]
        self.std = np.array(std_rgb, np.float32)[np.newaxis, np.newaxis, :]

    def __call__(self, img):
        dtype = get_dtype(None)
        img = img.astype(dtype)
        img *= 1.0 / 255.0

        img -= self.mean
        img /= self.std
        return img


class VOCSegTestTransform(object):
    """
    ImageNet-1K validation transform.
    """
    def __init__(self,
                 ds_metainfo,
                 mean_rgb=(0.485, 0.456, 0.406),
                 std_rgb=(0.229, 0.224, 0.225)):
        assert (ds_metainfo is not None)
        self.mean = np.array(mean_rgb, np.float32)[np.newaxis, np.newaxis, :]
        self.std = np.array(std_rgb, np.float32)[np.newaxis, np.newaxis, :]

    def __call__(self, img):
        dtype = get_dtype(None)
        img = img.astype(dtype)
        img *= 1.0 / 255.0

        img -= self.mean
        img /= self.std
        return img


class VOCMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(VOCMetaInfo, self).__init__()
        self.label = "VOC"
        self.short_label = "voc"
        self.root_dir_name = "voc"
        self.dataset_class = VOCSegDataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = VOCSegDataset.classes
        self.input_image_size = (480, 480)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_extra_kwargs = None
        self.test_metric_capts = ["Val.PixAcc", "Val.IoU"]
        self.test_metric_names = ["PixelAccuracyMetric", "MeanIoUMetric"]
        self.test_metric_extra_kwargs = [
            {"vague_idx": VOCSegDataset.vague_idx,
             "use_vague": VOCSegDataset.use_vague,
             "macro_average": False},
            {"num_classes": VOCSegDataset.classes,
             "vague_idx": VOCSegDataset.vague_idx,
             "use_vague": VOCSegDataset.use_vague,
             "bg_idx": VOCSegDataset.background_idx,
             "ignore_bg": VOCSegDataset.ignore_bg,
             "macro_average": False}]
        self.saver_acc_ind = 1
        self.train_transform = voc_train_transform
        self.val_transform = voc_val_transform
        self.test_transform = voc_val_transform
        self.train_transform2 = VOCSegTrainTransform
        self.val_transform2 = VOCSegTestTransform
        self.test_transform2 = VOCSegTestTransform
        self.train_generator = voc_train_generator
        self.val_generator = voc_val_generator
        self.test_generator = voc_test_generator
        self.ml_type = "imgseg"
        self.allow_hybridize = False
        self.net_extra_kwargs = {"aux": False, "fixed_size": False}
        self.load_ignore_extra = True
        self.image_base_size = 520
        self.image_crop_size = 480

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        super(VOCMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--image-base-size",
            type=int,
            default=520,
            help="base image size")
        parser.add_argument(
            "--image-crop-size",
            type=int,
            default=480,
            help="crop image size")

    def update(self,
               args):
        super(VOCMetaInfo, self).update(args)
        self.image_base_size = args.image_base_size
        self.image_crop_size = args.image_crop_size


def voc_train_transform(ds_metainfo,
                        data_format="channels_last"):
    """
    Create image transform sequence for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = SegImageDataGenerator(
        preprocessing_function=(lambda img: VOCSegTrainTransform(ds_metainfo=ds_metainfo)(img)),
        data_format=data_format)
    return data_generator


def voc_val_transform(ds_metainfo,
                      data_format="channels_last"):
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = SegImageDataGenerator(
        preprocessing_function=(lambda img: VOCSegTestTransform(ds_metainfo=ds_metainfo)(img)),
        data_format=data_format)
    return data_generator


def voc_train_generator(data_generator,
                        ds_metainfo,
                        batch_size):
    """
    Create image generator for training subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    split = "train"
    root = ds_metainfo.root_dir_path
    root = os.path.join(root, split)
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation=ds_metainfo.interpolation_msg,
        dataset=None)
    return generator


def voc_val_generator(data_generator,
                      ds_metainfo,
                      batch_size):
    """
    Create image generator for validation subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    split = "val"
    root = ds_metainfo.root_dir_path
    root = os.path.join(root, split)
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation="bilinear",
        dataset=ds_metainfo.dataset_class(
            root=ds_metainfo.root_dir_path,
            mode="val",
            transform=VOCSegTestTransform(
                ds_metainfo=ds_metainfo)))
    return generator


def voc_test_generator(data_generator,
                       ds_metainfo,
                       batch_size):
    """
    Create image generator for testing subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        Pascal VOC2012 dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    split = "val"
    root = ds_metainfo.root_dir_path
    root = os.path.join(root, split)
    generator = data_generator.flow_from_directory(
        directory=root,
        target_size=ds_metainfo.input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation="bilinear",
        dataset=ds_metainfo.dataset_class(
            root=ds_metainfo.root_dir_path,
            mode="test",
            transform=VOCSegTestTransform(
                ds_metainfo=ds_metainfo)))
    return generator
