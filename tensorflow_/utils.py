import tensorflow as tf

from .model_provider import get_model


def prepare_model(model_name,
                  classes,
                  use_pretrained):
    kwargs = {'pretrained': use_pretrained,
              'classes': classes}

    net = get_model(model_name, **kwargs)

    x = tf.placeholder(
        dtype=tf.float32,
        shape=(None, 3, 224, 224),
        name='xx')
    y_net = net(x)

    return y_net
