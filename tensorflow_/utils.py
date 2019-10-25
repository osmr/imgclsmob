import numpy as np
import tensorflow as tf

from .tensorflowcv.model_provider import get_model
from .tensorflowcv.models.common import is_channels_first


def save_model_params(sess,
                      file_path):
    # assert file_path.endswith('.npz')
    param_dict = {v.name: v.eval(sess) for v in tf.global_variables()}
    np.savez_compressed(file_path, **param_dict)


def load_model_params(net,
                      param_dict,
                      sess,
                      ignore_missing=False):
    for param_name, param_data in param_dict:
        with tf.variable_scope(param_name, reuse=True):
            try:
                var = tf.get_variable(param_name)
                sess.run(var.assign(param_data))
            except ValueError:
                if not ignore_missing:
                    raise


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path):
    data_format = "channels_first"
    kwargs = {"pretrained": use_pretrained, "data_format": data_format}

    net = get_model(model_name, **kwargs)
    input_image_size = net.in_size[0] if hasattr(net, 'in_size') else 224

    x_shape = (None, 3, input_image_size, input_image_size) if is_channels_first(data_format) else\
        (None, input_image_size, input_image_size, 3)
    x = tf.placeholder(
        dtype=tf.float32,
        shape=x_shape,
        name='xx')
    y_net = net(x)

    if use_pretrained or pretrained_model_file_path:
        from .tensorflowcv.model_provider import init_variables_from_state_dict
        with tf.Session() as sess:
            from .tensorflowcv.model_provider import load_state_dict
            if pretrained_model_file_path:
                init_variables_from_state_dict(
                    sess=sess,
                    state_dict=load_state_dict(file_path=pretrained_model_file_path))
            else:
                init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)

    return y_net
