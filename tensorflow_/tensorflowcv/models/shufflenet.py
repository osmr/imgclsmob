"""
    ShuffleNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.
"""

__all__ = ['ShuffleNet', 'shufflenet_g1_w1', 'shufflenet_g2_w1', 'shufflenet_g3_w1', 'shufflenet_g4_w1',
           'shufflenet_g8_w1', 'shufflenet_g1_w3d4', 'shufflenet_g3_w3d4', 'shufflenet_g1_wd2', 'shufflenet_g3_wd2',
           'shufflenet_g1_wd4', 'shufflenet_g3_wd4']

import os
import tensorflow as tf
from .common import conv1x1, conv3x3, depthwise_conv3x3, batchnorm, channel_shuffle, maxpool2d, avgpool2d,\
    is_channels_first, get_channel_axis, flatten


def shuffle_unit(x,
                 in_channels,
                 out_channels,
                 groups,
                 downsample,
                 ignore_group,
                 training,
                 data_format,
                 name="shuffle_unit"):
    """
    ShuffleNet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups in convolution layers.
    downsample : bool
        Whether do downsample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'shuffle_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 4

    if downsample:
        out_channels -= in_channels

    identity = x

    x = conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        groups=(1 if ignore_group else groups),
        data_format=data_format,
        name=name + "/compress_conv1")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/compress_bn1")
    x = tf.nn.relu(x, name=name + "/activ")

    x = channel_shuffle(
        x=x,
        groups=groups,
        data_format=data_format)

    x = depthwise_conv3x3(
        x=x,
        channels=mid_channels,
        strides=(2 if downsample else 1),
        data_format=data_format,
        name=name + "/dw_conv2")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/dw_bn2")

    x = conv1x1(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        groups=groups,
        data_format=data_format,
        name=name + "/expand_conv3")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/expand_bn3")

    if downsample:
        identity = avgpool2d(
            x=identity,
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name=name + "/avgpool")
        x = tf.concat([x, identity], axis=get_channel_axis(data_format), name=name + "/concat")
    else:
        x = x + identity

    x = tf.nn.relu(x, name=name + "/final_activ")
    return x


def shuffle_init_block(x,
                       in_channels,
                       out_channels,
                       training,
                       data_format,
                       name="shuffle_init_block"):
    """
    ShuffleNet specific initial block.

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
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'shuffle_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        data_format=data_format,
        name=name + "/conv")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/bn")
    x = tf.nn.relu(x, name=name + "/activ")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=1,
        data_format=data_format,
        name=name + "/pool")
    return x


class ShuffleNet(object):
    """
    ShuffleNet model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    groups : int
        Number of groups in convolution layers.
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
                 groups,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.groups = groups
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
        x = shuffle_init_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            training=training,
            data_format=self.data_format,
            name="features/init_block")
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                ignore_group = (i == 0) and (j == 0)
                x = shuffle_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=self.groups,
                    downsample=downsample,
                    ignore_group=ignore_group,
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


def get_shufflenet(groups,
                   width_scale,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".tensorflow", "models"),
                   **kwargs):
    """
    Create ShuffleNet model with specific parameters.

    Parameters:
    ----------
    groups : int
        Number of groups in convolution layers.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """

    init_block_channels = 24
    layers = [4, 8, 4]

    if groups == 1:
        channels_per_layers = [144, 288, 576]
    elif groups == 2:
        channels_per_layers = [200, 400, 800]
    elif groups == 3:
        channels_per_layers = [240, 480, 960]
    elif groups == 4:
        channels_per_layers = [272, 544, 1088]
    elif groups == 8:
        channels_per_layers = [384, 768, 1536]
    else:
        raise ValueError("The {} of groups is not supported".format(groups))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    net = ShuffleNet(
        channels=channels,
        init_block_channels=init_block_channels,
        groups=groups,
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


def shufflenet_g1_w1(**kwargs):
    """
    ShuffleNet 1x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=1, width_scale=1.0, model_name="shufflenet_g1_w1", **kwargs)


def shufflenet_g2_w1(**kwargs):
    """
    ShuffleNet 1x (g=2) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=2, width_scale=1.0, model_name="shufflenet_g2_w1", **kwargs)


def shufflenet_g3_w1(**kwargs):
    """
    ShuffleNet 1x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=3, width_scale=1.0, model_name="shufflenet_g3_w1", **kwargs)


def shufflenet_g4_w1(**kwargs):
    """
    ShuffleNet 1x (g=4) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=4, width_scale=1.0, model_name="shufflenet_g4_w1", **kwargs)


def shufflenet_g8_w1(**kwargs):
    """
    ShuffleNet 1x (g=8) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=8, width_scale=1.0, model_name="shufflenet_g8_w1", **kwargs)


def shufflenet_g1_w3d4(**kwargs):
    """
    ShuffleNet 0.75x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=1, width_scale=0.75, model_name="shufflenet_g1_w3d4", **kwargs)


def shufflenet_g3_w3d4(**kwargs):
    """
    ShuffleNet 0.75x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=3, width_scale=0.75, model_name="shufflenet_g3_w3d4", **kwargs)


def shufflenet_g1_wd2(**kwargs):
    """
    ShuffleNet 0.5x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=1, width_scale=0.5, model_name="shufflenet_g1_wd2", **kwargs)


def shufflenet_g3_wd2(**kwargs):
    """
    ShuffleNet 0.5x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=3, width_scale=0.5, model_name="shufflenet_g3_wd2", **kwargs)


def shufflenet_g1_wd4(**kwargs):
    """
    ShuffleNet 0.25x (g=1) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=1, width_scale=0.25, model_name="shufflenet_g1_wd4", **kwargs)


def shufflenet_g3_wd4(**kwargs):
    """
    ShuffleNet 0.25x (g=3) model from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
    Devices,' https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_shufflenet(groups=3, width_scale=0.25, model_name="shufflenet_g3_wd4", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        shufflenet_g1_w1,
        shufflenet_g2_w1,
        shufflenet_g3_w1,
        shufflenet_g4_w1,
        shufflenet_g8_w1,
        shufflenet_g1_w3d4,
        shufflenet_g3_w3d4,
        shufflenet_g1_wd2,
        shufflenet_g3_wd2,
        shufflenet_g1_wd4,
        shufflenet_g3_wd4,
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
        assert (model != shufflenet_g1_w1 or weight_count == 1531936)
        assert (model != shufflenet_g2_w1 or weight_count == 1733848)
        assert (model != shufflenet_g3_w1 or weight_count == 1865728)
        assert (model != shufflenet_g4_w1 or weight_count == 1968344)
        assert (model != shufflenet_g8_w1 or weight_count == 2434768)
        assert (model != shufflenet_g1_w3d4 or weight_count == 975214)
        assert (model != shufflenet_g3_w3d4 or weight_count == 1238266)
        assert (model != shufflenet_g1_wd2 or weight_count == 534484)
        assert (model != shufflenet_g3_wd2 or weight_count == 718324)
        assert (model != shufflenet_g1_wd4 or weight_count == 209746)
        assert (model != shufflenet_g3_wd4 or weight_count == 305902)

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
