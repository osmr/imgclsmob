"""
    ImageNet-1K classification dataset (via MXNet image record iterators).
"""

import os
import mxnet as mx
from .imagenet1k_cls_dataset import ImageNet1KMetaInfo, calc_val_resize_value


class ImageNet1KRecMetaInfo(ImageNet1KMetaInfo):
    def __init__(self):
        super(ImageNet1KRecMetaInfo, self).__init__()
        self.use_imgrec = True
        self.label = "ImageNet1K_rec"
        self.root_dir_name = "imagenet_rec"
        self.dataset_class = None
        self.num_training_samples = 1281167
        self.train_imgrec_file_path = "train.rec"
        self.train_imgidx_file_path = "train.idx"
        self.val_imgrec_file_path = "val.rec"
        self.val_imgidx_file_path = "val.idx"
        self.train_imgrec_iter = imagenet_train_imgrec_iter
        self.val_imgrec_iter = imagenet_val_imgrec_iter


def imagenet_train_imgrec_iter(ds_metainfo,
                               batch_size,
                               num_workers,
                               mean_rgb=(123.68, 116.779, 103.939),
                               std_rgb=(58.393, 57.12, 57.375),
                               jitter_param=0.4,
                               lighting_param=0.1):
    assert (isinstance(ds_metainfo.input_image_size, tuple) and len(ds_metainfo.input_image_size) == 2)
    imgrec_file_path = os.path.join(ds_metainfo.root_dir_path, ds_metainfo.train_imgrec_file_path)
    imgidx_file_path = os.path.join(ds_metainfo.root_dir_path, ds_metainfo.train_imgidx_file_path)
    data_shape = (ds_metainfo.in_channels,) + ds_metainfo.input_image_size
    kwargs = {
        "path_imgrec": imgrec_file_path,
        "path_imgidx": imgidx_file_path,
        "preprocess_threads": num_workers,
        "shuffle": True,
        "batch_size": batch_size,
        "data_shape": data_shape,
        "mean_r": mean_rgb[0],
        "mean_g": mean_rgb[1],
        "mean_b": mean_rgb[2],
        "std_r": std_rgb[0],
        "std_g": std_rgb[1],
        "std_b": std_rgb[2],
        "rand_mirror": True,
        "random_resized_crop": True,
        "max_aspect_ratio": (4.0 / 3.0),
        "min_aspect_ratio": (3.0 / 4.0),
        "max_random_area": 1,
        "min_random_area": 0.08,
        "brightness": jitter_param,
        "saturation": jitter_param,
        "contrast": jitter_param,
        "pca_noise": lighting_param
    }
    if ds_metainfo.aug_type == "aug0":
        pass
    elif ds_metainfo.aug_type == "aug1":
        kwargs["inter_method"] = 10
    elif ds_metainfo.aug_type == "aug2":
        kwargs["inter_method"] = 10
        kwargs["max_rotate_angle"] = 30
        kwargs["max_shear_ratio"] = 0.05
    else:
        raise RuntimeError("Unknown augmentation type: {}\n".format(ds_metainfo.aug_type))
    return mx.io.ImageRecordIter(**kwargs)


def imagenet_val_imgrec_iter(ds_metainfo,
                             batch_size,
                             num_workers,
                             mean_rgb=(123.68, 116.779, 103.939),
                             std_rgb=(58.393, 57.12, 57.375)):
    assert (isinstance(ds_metainfo.input_image_size, tuple) and len(ds_metainfo.input_image_size) == 2)
    imgrec_file_path = os.path.join(ds_metainfo.root_dir_path, ds_metainfo.val_imgrec_file_path)
    imgidx_file_path = os.path.join(ds_metainfo.root_dir_path, ds_metainfo.val_imgidx_file_path)
    data_shape = (ds_metainfo.in_channels,) + ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    return mx.io.ImageRecordIter(
        path_imgrec=imgrec_file_path,
        path_imgidx=imgidx_file_path,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,
        resize=resize_value,
        data_shape=data_shape,
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2])
