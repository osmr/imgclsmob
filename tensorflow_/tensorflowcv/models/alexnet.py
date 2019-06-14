"""
    AlexNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.
"""

__all__ = ['AlexNet', 'alexnet']

import os
import tensorflow as tf
from .common import conv2d, maxpool2d, is_channels_first, flatten


def alex_conv(x,
              in_channels,
              out_channels,
              kernel_size,
              strides,
              padding,
              data_format,
              name="alex_conv"):
    """
    AlexNet specific convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'alex_conv'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=True,
        data_format=data_format,
        name=name + "/conv")
    x = tf.nn.relu(x, name=name + "/activ")
    return x


def alex_dense(x,
               in_channels,
               out_channels,
               training,
               name="alex_dense"):
    """
    AlexNet specific dense block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    name : str, default 'alex_dense'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    assert (in_channels > 0)
    x = tf.layers.dense(
        inputs=x,
        units=out_channels,
        name=name + "/fc")
    x = tf.nn.relu(x, name=name + "/activ")
    x = tf.layers.dropout(
        inputs=x,
        rate=0.5,
        training=training,
        name=name + "/dropout")
    return x


def alex_output_block(x,
                      in_channels,
                      classes,
                      training,
                      name="alex_output_block"):
    """
    AlexNet specific output block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    name : str, default 'alex_output_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = 4096

    x = alex_dense(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        training=training,
        name=name + "/fc1")
    x = alex_dense(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        training=training,
        name=name + "/fc2")
    x = tf.layers.dense(
        inputs=x,
        units=classes,
        name=name + "/fc3")
    return x


class AlexNet(object):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    kernel_sizes : list of list of int
        Convolution window sizes for each unit.
    strides : list of list of int or tuple/list of 2 int
        Strides of the convolution for each unit.
    paddings : list of list of int or tuple/list of 2 int
        Padding value for convolution layer for each unit.
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
                 kernel_sizes,
                 strides,
                 paddings,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
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
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                x = alex_conv(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_sizes[i][j],
                    strides=self.strides[i][j],
                    padding=self.paddings[i][j],
                    data_format=self.data_format,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
            x = maxpool2d(
                x=x,
                pool_size=3,
                strides=2,
                padding=0,
                data_format=self.data_format,
                name="features/stage{}/pool".format(i + 1))

        in_channels = in_channels * 6 * 6
        x = flatten(
            x=x,
            data_format=self.data_format)
        x = alex_output_block(
            x=x,
            in_channels=in_channels,
            classes=self.classes,
            training=training,
            name="output")

        return x


def get_alexnet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create AlexNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    channels = [[64], [192], [384, 256, 256]]
    kernel_sizes = [[11], [5], [3, 3, 3]]
    strides = [[4], [1], [1, 1, 1]]
    paddings = [[2], [2], [1, 1, 1]]

    net = AlexNet(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
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


def alexnet(**kwargs):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_alexnet(model_name="alexnet", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        alexnet,
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
        assert (model != alexnet or weight_count == 61100840)

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
