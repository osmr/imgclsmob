import logging
import os
import tensorflow as tf
from .tensorflowcv2.model_provider import get_model


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  batch_size=None,
                  use_cuda=True):
    kwargs = {"pretrained": use_pretrained}
    # kwargs["input_shape"] = (1, 224, 224, 3)

    # my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    # tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
    # tf.debugging.set_log_device_placement(True)

    if not use_cuda:
        with tf.device("/cpu:0"):
            net = get_model(model_name, **kwargs)
            input_shape = ((1, 3, net.in_size[0], net.in_size[1]) if
                           net.data_format == "channels_first" else (1, net.in_size[0], net.in_size[1], 3))
            net.build(input_shape=input_shape)
    else:
        net = get_model(model_name, **kwargs)
        input_shape = ((batch_size, 3, net.in_size[0], net.in_size[1]) if
                       net.data_format == "channels_first" else (batch_size, net.in_size[0], net.in_size[1], 3))
        net.build(input_shape=input_shape)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))
        net.load_weights(filepath=pretrained_model_file_path)

    return net
