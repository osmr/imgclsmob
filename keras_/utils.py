import math
import logging
import os

from keras import backend as K
from keras.utils.np_utils import to_categorical
import mxnet as mx

from keras_.kerascv.model_provider import get_model


def prepare_ke_context(num_gpus,
                       batch_size):
    batch_size *= max(1, num_gpus)
    return batch_size


def get_data_rec(rec_train,
                 rec_train_idx,
                 rec_val,
                 rec_val_idx,
                 batch_size,
                 num_workers,
                 input_image_size=(224, 224),
                 resize_inv_factor=0.875,
                 only_val=False):
    assert (resize_inv_factor > 0.0)
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)

    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    data_shape = (3,) + input_image_size
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))

    if not only_val:
        train_data = mx.io.ImageRecordIter(
            path_imgrec=rec_train,
            path_imgidx=rec_train_idx,
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
            max_aspect_ratio=(4. / 3.),
            min_aspect_ratio=(3. / 4.),
            max_random_area=1,
            min_random_area=0.08,
            brightness=jitter_param,
            saturation=jitter_param,
            contrast=jitter_param,
            pca_noise=lighting_param,
        )
    else:
        train_data = None
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
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
        std_b=std_rgb[2],
    )
    return train_data, val_data


def get_data_generator(data_iterator,
                       num_classes):
    def get_arrays(db):
        data = db.data[0].asnumpy()
        if K.image_data_format() == "channels_last":
            data = data.transpose((0, 2, 3, 1))
        labels = to_categorical(
            y=db.label[0].asnumpy(),
            num_classes=num_classes)
        return data, labels

    while True:
        try:
            db = data_iterator.next()

        except StopIteration:
            # logging.warning("get_data exception due to end of data - resetting iterator")
            data_iterator.reset()
            db = data_iterator.next()

        finally:
            yield get_arrays(db)


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path):
    kwargs = {"pretrained": use_pretrained}

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))
        net.load_weights(filepath=pretrained_model_file_path)

    return net


def backend_agnostic_compile(model,
                             loss,
                             optimizer,
                             metrics,
                             num_gpus):
    keras_backend_exist = True
    try:
        K._backend
    except (NameError, AttributeError):
        keras_backend_exist = False
    if keras_backend_exist and (K._backend == "mxnet"):
        mx_ctx = ["gpu(%d)" % i for i in range(num_gpus)] if num_gpus > 0 else ["cpu()"]
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            context=mx_ctx)
    else:
        if num_gpus > 1:
            logging.info("Warning: num_gpus > 1 but not using MxNet backend")
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)
