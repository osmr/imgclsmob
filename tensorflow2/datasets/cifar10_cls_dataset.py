"""
    CIFAR-10 classification dataset.
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .dataset_metainfo import DatasetMetaInfo
from .cls_dataset import img_normalization


class CIFAR10MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CIFAR10MetaInfo, self).__init__()
        self.label = "CIFAR10"
        self.short_label = "cifar"
        self.root_dir_name = "cifar10"
        self.dataset_class = None
        self.num_training_samples = 50000
        self.in_channels = 3
        self.num_classes = 10
        self.input_image_size = (32, 32)
        self.train_metric_capts = ["Train.Err"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err"}]
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.saver_acc_ind = 0
        self.train_transform = cifar10_train_transform
        self.val_transform = cifar10_val_transform
        self.test_transform = cifar10_val_transform
        self.train_generator = cifar10_train_generator
        self.val_generator = cifar10_val_generator
        self.test_generator = cifar10_val_generator
        self.ml_type = "imgcls"
        self.mean_rgb = (0.4914, 0.4822, 0.4465)
        self.std_rgb = (0.2023, 0.1994, 0.2010)
        # self.interpolation_msg = "nearest"


def cifar10_train_transform(ds_metainfo,
                            data_format="channels_last"):
    """
    Create image transform sequence for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = ImageDataGenerator(
        preprocessing_function=(lambda img: img_normalization(
            img=img,
            mean_rgb=ds_metainfo.mean_rgb,
            std_rgb=ds_metainfo.std_rgb)),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format=data_format)
    return data_generator


def cifar10_val_transform(ds_metainfo,
                          data_format="channels_last"):
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    ImageDataGenerator
        Image transform sequence.
    """
    data_generator = ImageDataGenerator(
        preprocessing_function=(lambda img: img_normalization(
            img=img,
            mean_rgb=ds_metainfo.mean_rgb,
            std_rgb=ds_metainfo.std_rgb)),
        data_format=data_format)
    return data_generator


def cifar10_train_generator(data_generator,
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
    assert(ds_metainfo is not None)
    (x_train, y_train), _ = cifar10.load_data()
    generator = data_generator.flow(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=False)
    return generator


def cifar10_val_generator(data_generator,
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
    assert(ds_metainfo is not None)
    _, (x_test, y_test) = cifar10.load_data()
    generator = data_generator.flow(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        shuffle=False)
    return generator
