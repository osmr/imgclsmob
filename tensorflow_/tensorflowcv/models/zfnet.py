"""
    ZFNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.
"""

__all__ = ['zfnet', 'zfnetb']

import os
import tensorflow as tf
from .common import is_channels_first
from .alexnet import AlexNet


def get_zfnet(version="a",
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create ZFNet model with specific parameters.

    Parameters:
    ----------
    version : str, default 'a'
        Version of ZFNet ('a' or 'b').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if version == "a":
        channels = [[96], [256], [384, 384, 256]]
        kernel_sizes = [[7], [5], [3, 3, 3]]
        strides = [[2], [2], [1, 1, 1]]
        paddings = [[1], [0], [1, 1, 1]]
        use_lrn = True
    elif version == "b":
        channels = [[96], [256], [512, 1024, 512]]
        kernel_sizes = [[7], [5], [3, 3, 3]]
        strides = [[2], [2], [1, 1, 1]]
        paddings = [[1], [0], [1, 1, 1]]
        use_lrn = True
    else:
        raise ValueError("Unsupported ZFNet version {}".format(version))

    net = AlexNet(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        use_lrn=use_lrn,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_state_dict
        net.state_dict, net.file_path = download_state_dict(
            model_name=model_name,
            local_model_store_dir_path=root)
    else:
        net.state_dict = None
        net.file_path = None

    return net


def zfnet(**kwargs):
    """
    ZFNet model from 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_zfnet(model_name="zfnet", **kwargs)


def zfnetb(**kwargs):
    """
    ZFNet-b model from 'Visualizing and Understanding Convolutional Networks,' https://arxiv.org/abs/1311.2901.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_zfnet(version="b", model_name="zfnetb", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        zfnet,
        zfnetb,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)
        x = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224) if is_channels_first(data_format) else (None, 224, 224, 3),
            name="xx")
        y_net = net(x)

        weight_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != zfnet or weight_count == 62357608)
        assert (model != zfnetb or weight_count == 107627624)

        with tf.Session() as sess:
            if pretrained:
                from .model_store import init_variables_from_state_dict
                init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
            else:
                sess.run(tf.global_variables_initializer())
            x_value = np.zeros((1, 3, 224, 224) if is_channels_first(data_format) else (1, 224, 224, 3), np.float32)
            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
