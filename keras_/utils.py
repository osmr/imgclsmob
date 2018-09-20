import logging
import os

import keras
import mxnet as mx

from .model_provider import get_model


def prepare_ke_context(num_gpus,
                       batch_size):
    batch_size *= max(1, num_gpus)
    return batch_size


def get_data_rec(rec_train,
                 rec_train_idx,
                 rec_val,
                 rec_val_idx,
                 batch_size,
                 num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    train_data = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=num_workers,
        shuffle=True,
        batch_size=batch_size,

        data_shape=(3, 224, 224),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=(4. / 3.),
        min_aspect_ratio=(3. / 4.),
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,

        resize=256,
        data_shape=(3, 224, 224),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return train_data, val_data


def prepare_model(model_name,
                  classes,
                  use_pretrained,
                  pretrained_model_file_path):
    kwargs = {'pretrained': use_pretrained,
              'classes': classes}

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        net.load_weights(filepath=pretrained_model_file_path)

    return net


def backend_agnostic_compile(model,
                             loss,
                             optimizer,
                             metrics,
                             num_gpus):
    keras_backend_exist = True
    try:
        _ = keras.backend._backend
    except NameError:
        keras_backend_exist = False
    if keras_backend_exist and (keras.backend._backend == 'mxnet'):
        gpu_list = ["gpu(%d)" % i for i in range(num_gpus)]
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            context=gpu_list)
    else:
        if num_gpus > 1:
            print("Warning: num_gpus > 1 but not using MxNet backend")
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)
