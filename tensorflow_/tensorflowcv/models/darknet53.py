"""
    DarkNet-53 for ImageNet-1K, implemented in TensorFlow.
    Original source: 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.
"""

__all__ = ['DarkNet53', 'darknet53']

import os
import tensorflow as tf
from .common import conv1x1_block, conv3x3_block, is_channels_first, flatten


def dark_unit(x,
              in_channels,
              out_channels,
              alpha,
              training,
              data_format,
              name="dark_unit"):
    """
    DarkNet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    alpha : float
        Slope coefficient for Leaky ReLU activation.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'dark_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    assert (out_channels % 2 == 0)
    mid_channels = out_channels // 2

    identity = x
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        activation=(lambda y: tf.nn.leaky_relu(y, alpha=alpha, name=name + "/conv1/activ")),
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activation=(lambda y: tf.nn.leaky_relu(y, alpha=alpha, name=name + "/conv2/activ")),
        training=training,
        data_format=data_format,
        name=name + "/conv2")
    x = x + identity
    return x


class DarkNet53(object):
    """
    DarkNet-53 model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 alpha=0.1,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DarkNet53, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.alpha = alpha
        self.in_channels = in_channels
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

    def __call__(self,
                 x,
                 training=False):
        """
        Build a model graph.

        Parameters:
        ----------
        x : Tensor
            Input tensor.
        training : bool, or a TensorFlow boolean scalar tensor, default False
          Whether to return the output in training mode or in inference mode.

        Returns
        -------
        Tensor
            Resulted tensor.
        """
        in_channels = self.in_channels
        x = conv3x3_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            activation=(lambda y: tf.nn.leaky_relu(
                y,
                alpha=self.alpha,
                name="features/init_block/activ")),
            training=training,
            data_format=self.data_format,
            name="features/init_block")
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    x = conv3x3_block(
                        x=x,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=2,
                        activation=(lambda y: tf.nn.leaky_relu(
                            y,
                            alpha=self.alpha,
                            name="features/stage{}/unit{}/active".format(i + 1, j + 1))),
                        training=training,
                        data_format=self.data_format,
                        name="features/stage{}/unit{}".format(i + 1, j + 1))
                else:
                    x = dark_unit(
                        x=x,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        alpha=self.alpha,
                        training=training,
                        data_format=self.data_format,
                        name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
        x = tf.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=self.data_format,
            name="features/final_pool")(x)

        # x = tf.layers.flatten(x)
        x = flatten(
            x=x,
            data_format=self.data_format)
        x = tf.keras.layers.Dense(
            units=self.classes,
            name="output")(x)

        return x


def get_darknet53(model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
                  **kwargs):
    """
    Create DarkNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [2, 3, 9, 9, 5]
    channels_per_layers = [64, 128, 256, 512, 1024]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = DarkNet53(
        channels=channels,
        init_block_channels=init_block_channels,
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


def darknet53(**kwargs):
    """
    DarkNet-53 'Reference' model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_darknet53(model_name="darknet53", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        darknet53,
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
        assert (model != darknet53 or weight_count == 41609928)

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
