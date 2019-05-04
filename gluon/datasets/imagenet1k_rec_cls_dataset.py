"""
    ImageNet-1K classification dataset (via MXNet image record iterators).
"""

import mxnet as mx


def imagenet_train_imgrec_iter(imgrec_file_path,
                               imgidx_file_path,
                               batch_size,
                               num_workers,
                               data_shape=(3, 224, 224),
                               mean_rgb=(123.68, 116.779, 103.939),
                               std_rgb=(58.393, 57.12, 57.375),
                               jitter_param=0.4,
                               lighting_param=0.1):
    return mx.io.ImageRecordIter(
        path_imgrec=imgrec_file_path,
        path_imgidx=imgidx_file_path,
        preprocess_threads=num_workers,
        shuffle=True,
        batch_size=batch_size,
        data_shape=data_shape,
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=(4.0 / 3.0),
        min_aspect_ratio=(3.0 / 4.0),
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param)


def imagenet_val_imgrec_iter(imgrec_file_path,
                             imgidx_file_path,
                             batch_size,
                             num_workers,
                             data_shape=(3, 224, 224),
                             mean_rgb=(123.68, 116.779, 103.939),
                             std_rgb=(58.393, 57.12, 57.375),
                             resize_value=256):
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
