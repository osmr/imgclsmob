import numpy as np
import tensorflow as tf

from .tensorflowcv.model_provider import get_model


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
    kwargs = {'pretrained': use_pretrained}

    net = get_model(model_name, **kwargs)
    x = tf.placeholder(
        dtype=tf.float32,
        shape=(None, 3, 224, 224),
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
